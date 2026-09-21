"""Retrieval A/B between two recovery checkpoints.

The training loop reports perplexity only. Retrieval is the metric the surgery
actually damages -- linear attention replacing softmax, a 4x compressed KV
cache, no positional encoding are all long-range changes -- and it is measured
nowhere in the loop.

ALWAYS include the ORIGINAL arm. The question this project answers is whether a
converted model matches the model it was carved out of, and only that comparison
addresses it. Differences between two converted checkpoints describe adapter
configuration; treating one of them as the reference turns an internal ablation
into an apparent shortfall. Measured on longdoc, the converted model is ahead of
the original on perplexity AND retrieval at a 4x smaller KV cache -- a fact that
was obscured for some time by comparing converted models against each other.

Secondary question, and the reason for the data arms: every run before
2026-09-13 trained on a corpus whose median document is 550 tokens, sampled as
one concatenated stream, so an 8192-token window held about 15 unrelated
documents and no dependency longer than roughly 2k.

Reports gain = NLL(first occurrence) - NLL(second occurrence) at a set of gaps.
Higher is better: it is how much cheaper the needle becomes once the model has
already seen it, which is exactly what retrieval buys.
"""
import argparse
import json
import sys

import torch
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.surgery.transmla import convert_to_mla
from mercurius.adapters.lora import (inject_lora, freeze_base, inject_vera, unwrap_lora,
                  merge_and_restart)
from mercurius.models.kda import fuse_gate_lora
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.recovery.train import CKPT, EVAL_DATA
from mercurius.paths import CACHE_DIR, CKPT_DIR

GAPS = [1024, 4096, 16384]
LENGTHS = [2048, 8192]


def rules_from_checkpoint(sd):
    """Derive the LoRA rules from the checkpoint's own tensor shapes.

    The evaluator must NOT import LORA_RULES. That constant tracks whatever the
    current recipe uses, so raising the FFN rank to 32 immediately broke loading
    for every rank-16 checkpoint on disk -- strict=False forgives missing keys
    but not mismatched shapes, so it failed loudly rather than silently, which
    is the only reason this was cheap to find.

    A checkpoint records its own structure: each `<module>.lora_A` has shape
    (rank, in_features). Reading it back means old checkpoints stay loadable
    however the recipe moves.
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


def _fast_conv_for_stock(m):
    """Give the stock GDN layers of an ORIGINAL arm fla's Triton causal conv.

    Without the Dao-AILab causal_conv1d package (no sm_121 wheel) stock
    transformers falls back to a PyTorch depthwise conv, 5.7x slower here; the
    converted arms already use fla's kernel (models/kda.py). Same function, so
    the baseline is not slowed by a missing package. Outputs agree to bf16
    rounding (relL2 3.1e-3 on the conv alone).
    """
    from mercurius.models.kda import fla_causal_conv
    def conv(x, weight, bias=None, activation=None, seq_idx=None):
        return fla_causal_conv(x.transpose(1, 2), weight, bias,
                               activation=activation).transpose(1, 2)
    n = 0
    for mod in m.modules():
        if type(mod).__name__ == "Qwen3_5GatedDeltaNet" and mod.causal_conv1d_fn is None:
            mod.causal_conv1d_fn = conv
            n += 1
    return n


def build_original_nf4():
    """The unmodified original, NF4 with the trainer's quantization settings --
    the same-precision baseline for an NF4 student."""
    from mercurius.models.stream_nf4 import load_nf4
    from mercurius.recovery.train import ORIG
    m = load_nf4(ORIG, verbose=False).eval()
    _fast_conv_for_stock(m)
    return m


def build_original():
    """The unmodified teacher, no surgery and no adapters.

    This is the only comparison that settles whether the method works. Gaps
    between two converted models answer a question about adapter configuration,
    not about the result.
    """
    from transformers import AutoModelForCausalLM
    from mercurius.recovery.train import ORIG
    m = AutoModelForCausalLM.from_pretrained(ORIG, dtype=torch.bfloat16,
                                             device_map="cuda")
    _fast_conv_for_stock(m)
    return m.eval()


def build(adapters, dc, covs_path, double_adapter=False, init_adapters=None,
          alloc=None, quantize=False, merge_eval=False):
    """Reconstruct a trained model.

    double_adapter reproduces the pre-2026-09-13 injection, which wrapped every
    target twice. Checkpoints from that era carry both adapters per target, and
    a single injection leaves the inner ones unmatched -- strict=False drops
    them in silence and the model reads far worse than the run reported.
    """
    sd = torch.load(adapters, map_location="cpu")
    # A VeRA checkpoint records its own structure too: vera_d is (rank,) and the
    # module names say which targets were wrapped. Deriving it from the file
    # rather than from the recipe is the same rule as for LoRA -- the recipe
    # moves, the artifact does not.
    vera_d = {k for k in sd if k.endswith(".vera_d")}
    vera_rank = int(sd[next(iter(vera_d))].shape[0]) if vera_d else 0
    def _vera_pat(k):
        """Module pattern for a vera_d key, ignoring wrapper levels.

        A VeRA-wrapped projection that is LATER wrapped again -- per-head query
        maps wrap q_proj after VeRA does -- is named <...>.q_proj.base.vera_d.
        Taking the last two components then yields "q_proj.base", which matches
        nothing at injection time because the outer wrapper does not exist yet,
        so that adapter is never created and strict=False drops it in silence.
        rules_from_checkpoint already strips a trailing ".base" for LoRA for
        exactly this reason; this is the same normalisation for VeRA.
        """
        path = k[: -len(".vera_d")]
        while path.endswith(".base"):
            path = path[: -len(".base")]
        return ".".join(path.split(".")[-2:])
    vera_pats = sorted({_vera_pat(k) for k in vera_d})
    rules = rules_from_checkpoint(sd)
    by_r = {}
    for _, r in rules:
        by_r[r] = by_r.get(r, 0) + 1
    print(f"    ranks from checkpoint: "
          f"{', '.join(f'{c} pattern(s)@r{r}' for r, c in sorted(by_r.items()))}",
          flush=True)
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    trunk = m.model.language_model if hasattr(m.model, "language_model") else m.model
    for l in trunk.layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")
    # GDN-2 lift, if the checkpoint was trained on one. Detected from the
    # artifact: the two channel-wise gates only exist on a lifted layer, so
    # in_proj_be keys are proof the run did this and did it BEFORE adapters.
    #
    # The gate bases are not in the checkpoint -- the lift creates them, VeRA
    # freezes them, and they never pass through merge_and_restart so nothing tags
    # them as changed-and-frozen. They are recoverable because they are a
    # deterministic function of in_proj_b, which is untrained and comes straight
    # from stage-AB: row-tiling it reproduces them exactly.
    if any("in_proj_be" in k for k in sd):
        from mercurius.models.gdn2 import convert_to_gdn2
        n_lift = convert_to_gdn2(m, verbose=False)
        print(f"    replayed GDN-2 lift on {n_lift} layers "
              f"(gates row-tiled from in_proj_b)", flush=True)
    if vera_rank:
        # REPLAY THE INIT MERGE.
        #
        # A VeRA run does not start from stage-AB. It injects LoRA at the init
        # checkpoint's ranks, loads that checkpoint, folds the delta into the
        # bases, drops the wrappers, and only then installs VeRA. Everything it
        # folded in is frozen from that point on, so save_state -- which keys on
        # requires_grad -- does not record it. The checkpoint holds 156 vera_d /
        # vera_b pairs, 18 A_log, 18 dt_bias and 133 plain weights, and not one
        # base matrix for a VeRA target.
        #
        # That is recoverable rather than lost, because the missing weights are
        # a deterministic function of two files that are still on disk: stage-AB
        # plus the init delta. Replaying the startup sequence rebuilds them
        # exactly. Skipping the replay silently evaluates a model that never
        # existed -- it reads 22.609 instead of 17.861 -- and strict=False
        # reports nothing, because nothing is missing from the model's point of
        # view: the bases are simply the wrong (unmerged) values.
        #
        # The order below is the trainer's order, not a reconstruction of it.
        if init_adapters:
            isd = torch.load(init_adapters, map_location="cpu")
            irules = rules_from_checkpoint(isd)
            inject_lora(m, irules, verbose=False)
            freeze_base(m)
            m.load_state_dict({k: v.cuda() for k, v in isd.items()},
                              strict=False)
            merge_and_restart(m)
            # The run set every LoRA rule to rank 0 and removed the wrappers. A
            # checkpoint with no lora_A recorded that: it trained no LoRA at
            # all, so none survived the init. Unwrapping only the VeRA targets
            # would leave the others wrapped, and their weights then live at
            # <target>.base.weight, which no longer matches any checkpoint key.
            drop = ([pat for pat, _ in irules] + vera_pats) if not rules \
                   else list(vera_pats)
            n_un = unwrap_lora(m, drop)
            # fuse_gate_lora folds the in-class gate adapter into in_proj_a and
            # freezes it, so in_proj_a's weight is absent from the checkpoint
            # for the same reason the merged bases are. Detected from the
            # artifact: a run that did NOT fuse leaves a_lora_A/B trainable, and
            # they would appear here.
            fused = not any("a_lora_" in k for k in sd)
            if fused:
                fuse_gate_lora(m)
            merge_and_restart(m)
            print(f"    replayed init: {len(irules)} patterns from "
                  f"{init_adapters.split('/')[-1]}, merged, unwrapped {n_un}"
                  f"{', gate LoRA fused' if fused else ''}", flush=True)
        # vera_pats are PATTERNS, each matching many modules across layers.
        # Printing the pattern count as "targets" reads like the adapter covers
        # 11 modules when it covers 192.
        _nmod = len(vera_d)
        print(f"    VeRA checkpoint: rank {vera_rank}, {len(vera_pats)} patterns "
              f"covering {_nmod} modules", flush=True)
        inject_vera(m, [(p_, 1) for p_ in vera_pats], rank=vera_rank,
                    verbose=False)
    else:
        inject_lora(m, rules, verbose=False)
        if double_adapter:
            inject_lora(m, rules, verbose=False)
    freeze_base(m)
    if dc:
        covs = {int(k): v.cuda().float()
                for k, v in torch.load(covs_path, map_location="cpu").items()}
        convert_to_mla(m, d_c=(None if alloc else dc), alloc=alloc,
                       covs=covs, verbose=bool(alloc))
    # Per-head query maps, if the run had them. Detected from the artifact: the
    # R tensors exist only if install_per_head_q ran, and it runs last, after the
    # MLA conversion, so the rebuild must apply it at the same point.
    if any(k.endswith(".R") for k in sd):
        from mercurius.surgery.perhead_q import install_per_head_q
        install_per_head_q(m, verbose=False)
        print("    replayed per-head query maps", flush=True)

    # Honour the dtype the run used. The trainer promotes trainable norm gains to
    # fp32 (bf16 cannot hold them: zero-centered RMSNorm parks the two big groups
    # at exactly 0.0 and the three norms Stage A skipped overshoot), so 79 of
    # these tensors were saved as fp32. Loading them into bf16 parameters
    # truncates, and the rebuild then reads 0.15% off the run -- small enough to
    # look like noise on a machine where the eval has none.
    _params = dict(m.named_parameters())
    n_up = 0
    for k, v in sd.items():
        pm = _params.get(k)
        if pm is not None and v.dtype == torch.float32 and pm.dtype != torch.float32:
            pm.data = pm.data.float()
            n_up += 1
    if n_up:
        print(f"    restored {n_up} tensors to fp32 as the run held them",
              flush=True)
    missing = m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    unexpected = [k for k in sd if k not in dict(m.named_parameters())
                  and k not in dict(m.named_buffers())]
    if unexpected:
        print(f"    WARNING {len(unexpected)} checkpoint tensors had no home in "
              f"the model (e.g. {unexpected[0]}) -- the rebuild does not match "
              f"the run", flush=True)
    if quantize:
        # The trainer's last structural step (--student-bits 4): every FROZEN
        # Linear to NF4. "Frozen" in the run means "not in the checkpoint",
        # since the checkpoint is exactly the trained tensors; requires_grad
        # here is freeze_base's default set and does not match the run's, so it
        # cannot be the criterion. MLA latents, per-head R and norms are in the
        # file and stay high precision; VeRA-wrapped bases are not and get NF4.
        from mercurius.models.quantize import quantize_frozen_nf4
        for prm in m.parameters():
            prm.requires_grad_(False)
        named = dict(m.named_parameters())
        for k in sd:
            if k in named:
                named[k].requires_grad_(True)      # mark: keep high precision
        nq, nkeep = quantize_frozen_nf4(m)
        for prm in m.parameters():
            prm.requires_grad_(False)
        print(f"    student NF4: {nq} frozen Linear quantized, {nkeep} kept "
              f"high precision (as trained)", flush=True)
    if merge_eval:
        from mercurius.adapters.lora import merge_vera_for_eval
        nm = merge_vera_for_eval(m)
        torch.cuda.empty_cache()
        print(f"    merged {nm} VeRA adapters into bf16 bases for inference",
              flush=True)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="tag=path[:double] entries")
    ap.add_argument("--init-adapters",
                    default=str(CKPT_DIR / 'adapters-combined.pt'),
                    help="replayed to reconstruct merged base weights that "
                         "the checkpoint does not carry")
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--covs", default=str(CACHE_DIR / 'kv_covs.pt'))
    ap.add_argument("--out", default="logs/retrieval_ab.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(EVAL_DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    hdr = f"{'arm':<14}" + "".join(f"{'ppl@'+str(n):>10}" for n in LENGTHS) \
          + "".join(f"{'gain@'+str(k//1024)+'k':>11}" for k in GAPS)
    print(hdr); print("-" * len(hdr))
    rows = []
    for spec in a.arms:
        tag, rest = spec.split("=", 1)
        if rest == "ORIGINAL":
            m = build_original()
            row = {"arm": tag, "ppl": {}, "gain": {}, "nll1": {}, "nll2": {}}
            for n in LENGTHS:
                row["ppl"][n] = perplexity(m, ids, n); torch.cuda.empty_cache()
            for k in GAPS:
                # Keep BOTH occurrences, not only their difference. gain falls
                # when the needle's FIRST occurrence gets cheaper just as much as
                # when the second gets dearer, and those mean opposite things: the
                # first is better prediction, the second is worse retrieval. A
                # model that improves absolute NLL everywhere can post a lower
                # gain while retrieving no worse, and the difference alone cannot
                # tell the two apart.
                n1, n2, g = retrieval(m, ids, needle, k)
                row["gain"][k], row["nll1"][k], row["nll2"][k] = g, n1, n2
                torch.cuda.empty_cache()
            print(f"{tag:<14}" + "".join(f"{row['ppl'][n]:>10.3f}" for n in LENGTHS)
                  + "".join(f"{row['gain'][k]:>11.3f}" for k in GAPS), flush=True)
            rows.append(row); del m; torch.cuda.empty_cache()
            continue
        dbl = rest.endswith(":double")
        path = rest[:-7] if dbl else rest
        m = build(path, a.dc, a.covs, double_adapter=dbl,
                  init_adapters=a.init_adapters)
        row = {"arm": tag, "ppl": {}, "gain": {}, "nll1": {}, "nll2": {}}
        for n in LENGTHS:
            row["ppl"][n] = perplexity(m, ids, n)
            torch.cuda.empty_cache()
        for k in GAPS:
            n1, n2, g = retrieval(m, ids, needle, k)
            row["gain"][k], row["nll1"][k], row["nll2"][k] = g, n1, n2
            torch.cuda.empty_cache()
        print(f"{tag:<14}" + "".join(f"{row['ppl'][n]:>10.3f}" for n in LENGTHS)
              + "".join(f"{row['gain'][k]:>11.3f}" for k in GAPS), flush=True)
        rows.append(row)
        del m
        torch.cuda.empty_cache()

    if len(rows) > 1:
        base = rows[0]
        print("\n  vs the first arm (gain: higher is better):")
        for r in rows[1:]:
            dp = "".join(f"{(r['ppl'][n]/base['ppl'][n]-1)*100:>+9.2f}%" for n in LENGTHS)
            dg = "".join(f"{r['gain'][k]-base['gain'][k]:>+10.3f}" for k in GAPS)
            print(f"  {r['arm']:<14}{dp}{dg}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
