"""Minimal LoRA injection for nn.Linear.

Hand-rolled rather than peft: the KDA layer is a custom class carrying its own
gate LoRA, and wrapping it in peft complicates the save/load path we already
verified bit-exact. This is ~40 lines and keeps that path intact.

Zero-init B, so the adapted model starts exactly at the base model.
"""
import math
import torch
import torch.nn as nn


def _lora_scale(rank, alpha, rslora):
    if rslora:
        return (alpha if alpha is not None else 4.0) / math.sqrt(rank)
    return (alpha or rank) / rank


class LoRALinear(nn.Module):
    """out = base(x) + scale * (x @ A.T) @ B.T

    scale follows one of two conventions:

      classic (Hu et al.)   scale = alpha / rank
      rank-stabilized       scale = alpha / sqrt(rank)      (rsLoRA, 2312.03732)

    rsLoRA's Theorem 3.2: an adapter is rank-stabilized iff the factor is
    Theta(1/sqrt(r)). Under the classic convention with alpha tied to rank the
    factor is constant in r, so the adapter's output grows like sqrt(r) and rank
    is confounded with step size -- measured here, ||scale*BA|| grew as r^0.54
    after one step and r^0.73 by step 20. That makes a rank sweep uninterpretable
    and leaves the magnitude set by an arbitrary constant.

    With alpha fixed, alpha/sqrt(r) holds the output magnitude constant as rank
    changes. alpha=4 reproduces the previous scale of 1.0 at rank 16, so
    enabling rsLoRA at that alpha changes nothing at the current rank and only
    corrects the behaviour when rank moves.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None,
                 rslora: bool = False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.rslora = rslora
        self.scale = _lora_scale(rank, alpha, rslora)
        dev = base.weight.device
        dt = torch.bfloat16 if base.weight.dtype != torch.float32 else torch.float32
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        d = (x.to(self.lora_A.dtype) @ self.lora_A.T) @ self.lora_B.T
        return out + self.scale * d.to(out.dtype)


def rules_from_checkpoint(sd):
    """Derive LoRA rules from a checkpoint's own tensor shapes.

    Anything that rebuilds a model to match a checkpoint must read the
    CHECKPOINT, never the current recipe. LORA_RULES tracks whatever the recipe
    uses now, so every change to it invalidates loading for every checkpoint on
    disk -- raising the FFN rank from 16 to 32 broke both the evaluator and the
    --init-adapters path, in two separate places, on the same day.

    Each `<module>.lora_A` has shape (rank, in_features), so the structure is
    recoverable from the file.
    """
    rules = {}
    for k, v in sd.items():
        if not k.endswith(".lora_A"):
            continue
        path = k[: -len(".lora_A")]
        if path.endswith(".base"):          # inner adapter of a double wrap
            path = path[: -len(".base")]
        pat = ".".join(path.split(".")[-2:])
        rules[pat] = int(v.shape[0])
    return sorted(rules.items(), key=lambda kv: -len(kv[0]))


def inject_lora(model, rules, verbose=True, alpha=None, rslora=False):
    """rules: (pattern, rank) or (pattern, rank, alpha). First match wins; 0 skips.

    A PER-RULE alpha matters once the model has several rank groups. This one has
    three -- attention at 32, FFN at 32, KDA projections at whatever is being
    swept -- and a single global alpha under rsLoRA rescales all of them. That is
    not hypothetical: alpha=4 across mixed ranks left rank 16 alone but scaled
    the rank-32 attention adapters by 0.707, which changed the model at step 0
    and made two runs non-comparable.

    With a per-rule alpha each group keeps the scale it should, and rsLoRA's
    1/sqrt(r) law applies within the group whose rank is actually moving.
    """
    injected, total_new = [], 0
    seen = set()
    for mod_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if id(child) in seen:      # shared module reachable by two paths
                continue
            seen.add(id(child))
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            rank = None
            rule_alpha = alpha
            for rule in rules:
                pat, r = rule[0], rule[1]
                ra = rule[2] if len(rule) > 2 else None
                # endswith, NOT substring. A LoRA-wrapped target exposes its
                # frozen base at "<target>.base", which CONTAINS the pattern, so
                # a substring test re-wraps it on any second injection pass --
                # and train_recovery injects twice whenever --init-adapters is
                # used (once to shape the model for the load, once for real).
                # The result was two parallel rank-r adapters per target,
                # base(x) + lora_inner(x) + lora_outer(x): effectively rank 2r
                # at 2x the adapter params. Every run before 2026-09-13 has this.
                if full.endswith(pat):
                    rank = r
                    if ra is not None:
                        rule_alpha = ra
                    break
            if not rank:
                continue
            setattr(parent, child_name,
                    LoRALinear(child, rank, alpha=rule_alpha,
                               rslora=rslora and rule_alpha is not None))
            total_new += rank * (child.in_features + child.out_features)
            injected.append((full, rank))
    if verbose:
        by_rank = {}
        for name, r in injected:
            by_rank[r] = by_rank.get(r, 0) + 1
        print(f"  LoRA injected into {len(injected)} layers "
              f"({', '.join(f'{c}@r{r}' for r, c in sorted(by_rank.items()))}); "
              f"{total_new/1e6:.2f} M new params")
    return injected


def merge_and_restart(model, optimizer=None):
    """ReLoRA-style rank accumulation: fold each adapter into its base, then
    restart the adapter from zero.

    A rank-r adapter can only ever move the weight within an r-dimensional
    subspace. Merging releases that constraint: after k merges the accumulated
    update can reach rank k*r while never holding more than r(m+n) trainable
    parameters at once. Measured on this model, the dense update needs rank ~300
    on in_proj_qkv, which costs 2.1 M as a single factorization but only
    r(m+n) at a time this way.

    Two details that make or break it:

    1. THE OPTIMIZER STATE MUST BE DROPPED for the restarted factors. Adam's
       moments encode the direction the old subspace was moving in; carried
       across a merge they immediately drag the freshly initialized factors back
       into the subspace we just escaped, which is the whole point of merging.

    2. THE MERGED BASE MUST BE SAVED. base.weight is frozen, so a checkpointer
       keyed on requires_grad -- which is what save_trainable does, deliberately
       -- will not write it, and every merged update is silently lost at save
       time. This is the third instance of that failure mode in this project
       (the hardcoded name filter dropped in_proj_a, the substring match
       double-wrapped adapters), so modules are tagged here and the saver reads
       the tag rather than inferring anything.
    """
    n = 0
    for mod in model.modules():
        if not isinstance(mod, LoRALinear):
            continue
        with torch.no_grad():
            delta = (mod.lora_B.float() @ mod.lora_A.float()) * mod.scale
            mod.base.weight.data += delta.to(mod.base.weight.dtype)
            nn.init.kaiming_uniform_(mod.lora_A, a=math.sqrt(5))
            mod.lora_B.zero_()
        mod.merged = True                      # read by save_trainable
        if optimizer is not None:
            for p in (mod.lora_A, mod.lora_B):
                optimizer.state.pop(p, None)
        n += 1
    return n


def resize_lora(model, rules, verbose=True):
    """Give each adapter the rank its rule now asks for, discarding the old one.

    Only safe AFTER merge_and_restart has folded the existing adapters into
    their bases -- otherwise the delta they carry is thrown away. That ordering
    is what lets --init-adapters (rank 16 on disk) be combined with a different
    --kda-rank: load at the rank the checkpoint has, fold it in, then resize.
    Loading a rank-16 checkpoint into rank-64 adapters fails outright, since
    strict=False forgives missing keys but not mismatched shapes.
    """
    changed = []
    for name, mod in model.named_modules():
        if not isinstance(mod, LoRALinear):
            continue
        for rule in rules:
            pat, r = rule[0], rule[1]
            if not name.endswith(pat):
                continue
            if r and r != mod.rank:
                dev, dt = mod.lora_A.device, mod.lora_A.dtype
                mod.rank = r
                mod.scale = _lora_scale(r, mod.alpha, mod.rslora)
                mod.lora_A = nn.Parameter(torch.zeros(
                    r, mod.base.in_features, device=dev, dtype=dt))
                mod.lora_B = nn.Parameter(torch.zeros(
                    mod.base.out_features, r, device=dev, dtype=dt))
                nn.init.kaiming_uniform_(mod.lora_A, a=math.sqrt(5))
                changed.append((name, r))
            break
    if verbose and changed:
        by_r = {}
        for _, r in changed:
            by_r[r] = by_r.get(r, 0) + 1
        print(f"  resized {len(changed)} adapters "
              f"({', '.join(f'{c}@r{r}' for r, c in sorted(by_r.items()))})",
              flush=True)
    return len(changed)


def merged_base_names(model):
    """Parameter names of bases that absorbed a merge and must be checkpointed.

    Covers both shapes: a LoRALinear still holding its merged base, and a bare
    module that absorbed one and was then unwrapped (and possibly re-wrapped by
    a different adapter family). The tag travels with the weight, not with the
    wrapper, because the wrapper does not survive an adapter swap.
    """
    out = set()
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear) and getattr(mod, "merged", False):
            out.add(f"{name}.base.weight")
        elif getattr(mod, "_absorbed_merge", False):
            out.add(f"{name}.weight")
    return out


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def freeze_base(model, also_train=("lora_A", "lora_B", "a_lora_A", "a_lora_B",
                                   "A_log", "dt_bias", "pe_c")):
    """Freeze everything, then re-enable adapters and the gate parameters.

    pe_c is here because the decay-tied phase is created BEFORE this runs, so
    without it the parameter is frozen, never trains, and never appears in a
    checkpoint keyed on requires_grad. The phase33 run trained a FIXED c = 1e-2
    perturbation for 150 steps and reported it as a test of a learned phase.
    Any new trainable created before freeze_base needs an entry here.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, p in model.named_parameters():
        if any(k in name for k in also_train):
            p.requires_grad_(True)
            n += p.numel()
    return n


class VeRALinear(nn.Module):
    """VeRA (Kopiczko, Blankevoort, Asano, ICLR 2024; arXiv:2310.11454).

        dW = diag(b) . B . diag(d) . A

    A and B are RANDOM, FROZEN and SHARED ACROSS LAYERS; only the two vectors
    d (length rank) and b (length out_features) are learned, so a layer costs
    rank + out_features trainable parameters instead of rank*(in+out).

    Rank is therefore nearly free -- going from 16 to 256 adds 240 parameters
    per layer, not 240*(in+out) -- which is why VeRA is normally run at a rank
    far above anything sensible for LoRA.

    b is initialized to ZERO so dW = 0 exactly at the start, matching LoRA's
    zero-init B. d is initialized to a constant (the paper's d_init).
    """

    def __init__(self, base: nn.Linear, rank: int, shared_A, shared_B,
                 d_init: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.vera_A = shared_A          # (rank, max_in)   frozen, shared
        self.vera_B = shared_B          # (max_out, rank)  frozen, shared
        dev = base.weight.device
        dt = torch.bfloat16 if base.weight.dtype != torch.float32 else torch.float32
        self.vera_d = nn.Parameter(torch.full((rank,), d_init, device=dev, dtype=dt))
        self.vera_b = nn.Parameter(torch.zeros(base.out_features, device=dev, dtype=dt))

    def forward(self, x):
        out = self.base(x)
        I, O = self.base.in_features, self.base.out_features
        # diag(b) B diag(d) A, with the two diagonals folded into the SMALL
        # shared factors before the matmuls: (x (dA)^T)(bB)^T. Same function;
        # the scalings then cost rank*in + out*rank elementwise instead of
        # T*rank + T*out over the activations. Measured at 32k tokens the
        # unfolded form spent more time than the NF4 base matmuls it adapts
        # (4.28 s vs 2.94 s over 256 modules). Gradients reach d and b through
        # the folded products exactly as before.
        A = self.vera_A[:, :I] * self.vera_d.unsqueeze(1)      # (rank, in)
        B = self.vera_B[:O, :] * self.vera_b.unsqueeze(1)      # (out, rank)
        h = (x.to(A.dtype) @ A.T) @ B.T
        return out + h.to(out.dtype)


def inject_vera(model, rules, rank=256, d_init=0.1, verbose=True, seed=0):
    """Wrap matching Linears with VeRA, sharing ONE frozen random pair.

    Sharing is the point, not an optimization: it is what makes the per-layer
    cost rank + out_features. The shared pair is sized to the largest target and
    sliced per layer, which is how the paper handles differing shapes.
    """
    targets = []
    seen = set()
    for mod_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or id(child) in seen:
                continue
            seen.add(id(child))
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            for rule in rules:
                if full.endswith(rule[0]) and rule[1]:
                    targets.append((parent, child_name, child, full))
                    break
    if not targets:
        return []
    max_in = max(c.in_features for _, _, c, _ in targets)
    max_out = max(c.out_features for _, _, c, _ in targets)
    ref = targets[0][2].weight
    dt = torch.bfloat16 if ref.dtype != torch.float32 else torch.float32
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = nn.Parameter(torch.empty(rank, max_in, dtype=torch.float32), requires_grad=False)
    B = nn.Parameter(torch.empty(max_out, rank, dtype=torch.float32), requires_grad=False)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
    nn.init.kaiming_uniform_(B, a=math.sqrt(5), generator=g)
    A = A.to(ref.device, dt); B = B.to(ref.device, dt)
    A.requires_grad_(False); B.requires_grad_(False)

    for parent, child_name, child, _ in targets:
        setattr(parent, child_name, VeRALinear(child, rank, A, B, d_init))
    if verbose:
        per = sum(rank + c.out_features for _, _, c, _ in targets)
        print(f"  VeRA injected into {len(targets)} layers at rank {rank}; "
              f"{per/1e6:.3f} M trainable (shared A {tuple(A.shape)}, "
              f"B {tuple(B.shape)} frozen)", flush=True)
    return targets


def merge_vera(model, optimizer=None, reseed=True, seed=0, d_init=0.1,
               verbose=True):
    """Fold each VeRA adapter into its base, then RE-DRAW the shared A and B.

        dW = diag(b) . B[:O] . diag(d) . A[:I]

    The re-draw is the point, and it is what makes this different from ReLoRA on
    LoRA. VeRA's parameter count is not its constraint -- a layer holds only d
    (rank) and b (out_features) either way. Its constraint is that A and B are a
    FIXED random subspace: the update can only rescale directions it was handed,
    and no amount of training escapes that span. Merging and drawing a NEW A, B
    gives the next cycle an independent subspace, so k cycles accumulate an
    effective rank of k*rank while never holding more than one cycle's
    parameters. That attacks the binding constraint rather than the nominal one.

    Two things carried over from merge_and_restart, for the same reasons:

    1. THE OPTIMIZER STATE MUST BE DROPPED for d and b. Adam's moments encode
       motion within the old random subspace; carried across a re-draw they pull
       the new vectors back toward directions that no longer exist.

    2. THE MERGED BASE MUST BE SAVED. It is frozen, so a checkpointer keyed on
       requires_grad will not write it. Unlike the init merge, this one is NOT
       replayable -- it depends on the trained d and b at the moment of merging,
       which are then reset -- so a run that merges and saves adapters only
       produces a checkpoint that cannot be rebuilt at all. train_recovery
       refuses that combination rather than discovering it at eval time.
    """
    mods = [m for m in model.modules() if isinstance(m, VeRALinear)]
    if not mods:
        return 0
    with torch.no_grad():
        for mod in mods:
            I, O = mod.base.in_features, mod.base.out_features
            A = mod.vera_A[:, :I].float()
            B = mod.vera_B[:O, :].float()
            dW = mod.vera_b.float().unsqueeze(1) * ((B * mod.vera_d.float()) @ A)
            mod.base.weight.data += dW.to(mod.base.weight.dtype)
            mod.base._absorbed_merge = True
            mod.vera_b.zero_()
            mod.vera_d.fill_(d_init)
            if optimizer is not None:
                for p_ in (mod.vera_b, mod.vera_d):
                    optimizer.state.pop(p_, None)
        if reseed:
            # A and B are shared, so redraw each distinct tensor once
            seen = set()
            for mod in mods:
                for tensor in (mod.vera_A, mod.vera_B):
                    if id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    buf = torch.empty(tensor.shape, dtype=torch.float32)
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    nn.init.kaiming_uniform_(buf, a=math.sqrt(5), generator=g)
                    tensor.data.copy_(buf.to(tensor.device, tensor.dtype))
                    seed += 1
    if verbose:
        print(f"  VeRA merged into {len(mods)} bases"
              f"{' and A/B re-drawn' if reseed else ''}", flush=True)
    return len(mods)


def unwrap_lora(model, patterns):
    """Replace matching LoRALinear wrappers with their base module.

    Only safe after merge_and_restart, which folds each adapter's delta into its
    base -- otherwise the delta is discarded. Used to swap one adapter family
    for another on a subset of modules without losing the initialization they
    were carrying.
    """
    n = 0
    for mod_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if any(full.endswith(p) for p in patterns):
                # Carry the merge tag onto the base. Unwrapping destroys the
                # LoRALinear that merged_base_names looks for, so without this
                # the folded delta is trained, evaluated, and then silently
                # dropped at save time -- which is exactly what merge_and_restart
                # documents as the failure to avoid, reintroduced one layer up.
                if getattr(child, "merged", False):
                    child.base._absorbed_merge = True
                setattr(parent, child_name, child.base)
                n += 1
    return n


@torch.no_grad()
def merge_vera_for_eval(model, dtype=torch.bfloat16):
    """INFERENCE ONLY: replace every VeRALinear by one dense Linear holding
    base + diag(b) B diag(d) A, dequantizing an NF4 base first.

    At rank 1024 the adapter's two low-rank matmuls add ~55% of the base
    layer's FLOPs, and the NF4 path adds a dequantization per call; merged, each
    layer is a single cuBLAS matmul. The merged weight is rounded to bf16 once,
    so outputs differ from the training-time path at bf16-rounding level.
    Not for training: the merged layer has no adapter to train.
    Returns the number of layers merged.
    """
    import bitsandbytes as bnb
    n = 0
    for mod_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, VeRALinear):
                continue
            base = child.base
            if isinstance(base, bnb.nn.Linear4bit):
                W = bnb.functional.dequantize_4bit(
                    base.weight.data, base.weight.quant_state).float()
            else:
                W = base.weight.data.float()
            I, O = base.in_features, base.out_features
            A = child.vera_A[:, :I].float() * child.vera_d.float().unsqueeze(1)
            B = child.vera_B[:O, :].float() * child.vera_b.float().unsqueeze(1)
            W += B @ A
            lin = nn.Linear(I, O, bias=base.bias is not None,
                            device=W.device, dtype=dtype)
            lin.weight.copy_(W.to(dtype))
            if base.bias is not None:
                lin.bias.copy_(base.bias.to(dtype))
            lin.requires_grad_(False)
            setattr(parent, child_name, lin)
            n += 1
    return n
