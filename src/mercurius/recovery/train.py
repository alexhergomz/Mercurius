"""Stage C recovery: distil the converted+dialled model back toward the original.

Objective is KL to the FROZEN ORIGINAL model, not cross-entropy to text. The
budget's job is to undo conversion damage, and matching the teacher's
distribution is a far denser signal than next-token labels -- which matters when
the budget is ~10^8 tokens rather than 10^10.

Run in bf16, not NF4. Measured: NF4 costs +4.9..7.1% perplexity on this 0.8B
(embeddings are 32% of its parameters and are skipped as tied). That is the same
order as the NoPE damage we are trying to measure, and it will not transfer to
9B where embeddings are 2.8% of parameters. Quantization is validated as a
separate axis.

Trains on FineWeb-Edu, evaluates on WikiText-2. Separate corpora: an earlier
version sampled training windows from the same tokens perplexity was measured on,
which would have reported memorization as recovery.

Baselines this is measured against (from characterize.py, wikitext, same lengths):
    tiled  C0 full RoPE  : 12.820 / 18.320 / 14.530
    tiled  C3 NoPE       : 14.278 / 21.625 / 17.385
    seeded C3 NoPE       : 14.816 / 22.088 / 16.683
"""
import sys, os, json, time, argparse, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius.models.kda import (load_kda_model, enable_state_passing, reset_state,
                       promote_state)
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import (inject_lora, freeze_base, trainable_parameters,
                  merge_and_restart, merged_base_names)
from mercurius.eval.characterize import perplexity
from mercurius.eval.suite import ce_and_topk, sample, report, PROMPTS
import bitsandbytes as bnb
from mercurius.recovery.logit_cache import build_cache, load_cache, topk_kl, taid_kl
from mercurius.surgery.transmla import convert_to_mla
from mercurius.adapters.layerscale import install_layerscale
from mercurius.paths import BASE_MODEL, FINEWEB, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
ORIG = str(BASE_MODEL)
TRAIN_DATA = str(FINEWEB)   # train
EVAL_DATA  = str(WIKITEXT)      # eval, held out

LORA_RULES = [
    ("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
    ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),   # the dialled layers
    ("linear_attn.out_proj", 16), ("linear_attn.in_proj_qkv", 16),
    # in_proj_z is 16.5% of KDA and had no adaptation path at all: frozen under
    # every LoRA-only arm, dense only under --train-attn. Under --train-attn the
    # base unfreezes alongside this adapter, so dense runs gain 0.88 M redundant
    # adapter params -- val-full's recorded numbers predate this rule.
    ("linear_attn.in_proj_z", 16),
    ("mlp.gate_proj", 16), ("mlp.up_proj", 16), ("mlp.down_proj", 16),
    ("lm_head", 0), ("embed_tokens", 0),
]


def kl_loss(student_logits, teacher_logits, chunk=512, temperature=1.0):
    """Chunked forward KL(teacher || student). Vocab is 248,320: a full-sequence
    fp32 softmax would allocate tens of GiB."""
    T = student_logits.shape[0]
    tot, n = 0.0, 0
    loss = 0.0
    for i in range(0, T, chunk):
        s = student_logits[i:i + chunk].float() / temperature
        t = teacher_logits[i:i + chunk].float() / temperature
        lp_s = F.log_softmax(s, -1)
        p_t = F.softmax(t, -1)
        loss = loss + (p_t * (F.log_softmax(t, -1) - lp_s)).sum(-1).sum()
        n += s.shape[0]
    return loss / max(n, 1)


from torch.utils.checkpoint import checkpoint


def save_trainable(model, path):
    """Save exactly the parameters that were trained.

    The previous filter was a hardcoded name tuple, which goes stale the moment
    a new flag makes something else trainable: --train-gate unfreezes
    in_proj_a.weight (37.7M params) and --train-norms unfreezes the norm gains,
    and NEITHER matches ("lora_A","lora_B","A_log","dt_bias"). The run would
    train them, the eval curve would show the benefit, and the weights would be
    silently dropped on save -- the same measure-it-and-lose-it failure that
    per-eval checkpointing was added to prevent, one level down.

    Keying off requires_grad cannot go stale.
    """
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    # A merged base is frozen but CHANGED -- requires_grad cannot see it.
    names |= merged_base_names(model)
    sd = model.state_dict()
    torch.save({k: sd[k].detach().cpu() for k in sorted(names) if k in sd}, path)
    return len(names)


def _chunk_kl_terms(h_c, W_lm, tv_c, ti_c, lam, use_taid):
    """Per-position KL for one chunk of positions, never forming full logits.

    topk_kl reads only the k columns named by the teacher's top-K indices, and
    renormalizes over those same columns -- the other 248,256 are computed and
    discarded. Gathering the lm_head ROWS for those indices instead costs
    C*k*d FLOPs rather than C*V*d (~3,900x fewer at k=64, V=248,320) and bounds
    peak memory by the chunk instead of the sequence.

    This is what makes the 32k draws in the length mix affordable. Materializing
    the logits needs 15.16 GiB at 32k for the student alone, plus a gradient
    buffer of the same shape -- which is the exact allocation that OOMed in
    backward before this existed.
    """
    from mercurius.recovery.logit_cache import topk_kl_terms, taid_kl_terms
    s_sel = torch.einsum("cd,ckd->ck", h_c, W_lm[ti_c.long()].to(h_c.dtype))
    return (taid_kl_terms(s_sel, tv_c, lam) if use_taid
            else topk_kl_terms(s_sel, tv_c))


def _chunk_fullkl_terms(h_s_c, h_t_c, W_lm):
    """Per-position FULL-VOCABULARY KL(teacher || student) for one chunk.

    No top-k anywhere. Both C x V logit blocks are formed, reduced, and dropped
    -- the same containment trick as _chunk_kl_terms, but with a live teacher
    there is no cached support to truncate, so the estimator is exact.

    Why this matters more than it looks. Measured in the literature at K=50 on a
    248k-class vocabulary, vanilla top-K distillation closes about 5% of the gap
    between plain CE and full distillation, and its parameter gradient sits ~48
    degrees off the full-distillation gradient with 1.8x the norm. At K=12 it is
    WORSE than not distilling at all. Our cached runs were all K=64.

    Two structural holes disappear here as a side effect:
      * the ~10.8% of positions where the true next token fell outside the
        cached top-64 and the loss was simply silent about it;
      * topk_kl's invariance to total mass on the support -- it renormalizes
        both sides over the same 64 columns, so a student placing 1% of its mass
        there and 99% on garbage scored zero loss if the shape matched.
    """
    lg_t = (h_t_c @ W_lm.T).float()
    t_lp = F.log_softmax(lg_t, -1)
    del lg_t
    lg_s = (h_s_c @ W_lm.T).float()
    s_lp = F.log_softmax(lg_s, -1)
    del lg_s
    return (t_lp.exp() * (t_lp - s_lp)).sum(-1)


def _chunk_ce_terms(h_c, W_lm, tgt_c):
    """Teacher-free CE for one chunk.

    Unlike the KL this genuinely needs the full-vocabulary logsumexp, so it does
    form C x V -- bounded by head_chunk, and under checkpoint it is recomputed
    in backward rather than stored.
    """
    return F.cross_entropy((h_c @ W_lm.T).float(), tgt_c, reduction="none")


def save_resume(path, model, opt, sched, step, gens):
    """Everything needed to continue a run exactly where it stopped.

    Weight checkpoints alone are NOT resumable: restarting from them re-inits the
    optimizer moments and restarts OneCycleLR from step 0, so the run continues on
    a different trajectory. This session lost ~100 minutes of a 300-step run twice
    because a mid-flight environment change (a newly built kernel, a stale guard)
    could only be adopted by starting over.

    Written once per eval and OVERWRITTEN, so it costs a constant ~1.1 GiB rather
    than growing with the number of evals.
    """
    torch.save({"step": step,
                "weights": {n: p.detach().cpu() for n, p in model.named_parameters()
                            if p.requires_grad},
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "gens": {k: g.get_state() for k, g in gens.items()}}, path)


def batches(ids, seq_len_fn, steps, seed=0, align=1, gen=None):
    """Random windows -- each step independent.

    `align` snaps offsets to multiples of the teacher cache's block size. Without
    it a training window straddles teacher blocks, and neither side of the
    straddle is the conditional the loss assumes: positions early in a block get
    a teacher that saw LESS context than the student, and crossing into the next
    block gets a teacher whose context was reset mid-window. Aligning the offset
    and keeping the window no longer than one block makes teacher and student
    condition on exactly the same prefix.
    """
    g = gen if gen is not None else torch.Generator().manual_seed(seed)
    for step in range(steps):
        seq_len = seq_len_fn(step)
        hi = len(ids) - seq_len - 1
        i = int(torch.randint(0, hi, (1,), generator=g))
        if align > 1:
            i = (i // align) * align
        yield ids[i:i + seq_len].unsqueeze(0), False, i


# Variable-length sampling (Dataset Decomposition, 2405.13226): instead of a
# fixed length -- or a ramp that ends long and stops showing short sequences --
# draw a length per step from a mixture. Expensive long steps are amortized
# against cheap short ones, and the model keeps seeing both regimes, which the
# paper finds necessary for generalization. Unusually cheap for us: 18 of 24
# layers are linear, so only the 6 attention layers pay quadratic cost.
# Was [(1024,.35),(2048,.30),(8192,.25),(32768,.10)] -- 65% of draws at <=2048.
# Those windows cannot contain the dependency we are trying to teach: the
# measured compression damage is at 4k-16k GAPS, and a 2048-token window has no
# room for one. The matched control shows short-context behaviour was never
# damaged, so that 65% was spending compute repairing something that was not
# broken.
#
# The original rationale was that long steps are expensive. That is true when
# compute-bound and false here: profiling shows 82% of a step is CPU dispatch
# (3,433 as_strided + 2,694 view + 2,415 reshape per step), and launch count is
# set by layer/op count rather than sequence length. So a 32k step issues about
# as many kernels as a 2k step while doing 16x the work -- long windows are
# close to free per token in this regime. We never checked which regime we were
# in before choosing the mix.
LENGTH_MIX = [(8192, 0.90), (32768, 0.10)]
# Measured cost per step: 8192 un-checkpointed runs at 1249 tok/s (6.6 s/step);
# 32768 OOMs without checkpointing and manages only 161 tok/s WITH it (203
# s/step), because attention is quadratic and the 18 KDA layers' activations
# force full recompute. A 50/50 mix therefore costs ~105 s/step -- 11.6 h for
# 400 steps. At 90/10 it is 26 s/step, 2.9 h.
#
# Note this restores the ORIGINAL 32768 weight of 0.10. That part of the old mix
# was defensible; its error was the 65% of draws at <=2048, which cannot contain
# a 4k-16k gap and so trained a capability the control shows was never damaged.


def sample_length(gen):
    r = float(torch.rand(1, generator=gen))
    acc = 0.0
    for n, p in LENGTH_MIX:
        acc += p
        if r <= acc:
            return n
    return LENGTH_MIX[-1][0]


def mix_expected_tokens():
    return sum(n * p for n, p in LENGTH_MIX)


def seq_len_at(step, total, base, max_seq, ramp):
    """Stepped ramp over a FIXED SET of lengths (kept for A/B against the mix).

    A continuous ramp is pathological here: every distinct length is a new
    Triton kernel shape, and a cold shape costs up to ~2 min of autotune on this
    board. Powers of two give the same curriculum for four compilations.
    """
    if not ramp or max_seq <= base:
        return base
    stages = []
    n = base
    while n <= max_seq:
        stages.append(n)
        n *= 2
    frac = min(1.0, step / max(1.0, 0.75 * total))
    idx = min(len(stages) - 1, int(frac * len(stages)))
    return stages[idx]


def sequential_batches(ids, seq_len_fn, steps, stride_docs=64):
    """Contiguous segments, so carried state is genuinely the state that arises
    deep in a long context. Yields (batch, is_document_start)."""
    pos, seg = 0, 0
    for step in range(steps):
        seq_len = seq_len_fn(step)
        if pos + seq_len + 1 > len(ids):
            pos, seg = 0, 0
        new_doc = (seg % stride_docs == 0)
        yield ids[pos:pos + seq_len].unsqueeze(0), new_doc, pos
        pos += seq_len
        seg += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", default="nope", choices=["nope", "c1", "c0"])
    ap.add_argument("--seed-decay", action="store_true")
    ap.add_argument("--seed-alpha", type=float, default=0.60,
                    help="median alpha the decay seed targets; <=0 disables the "
                         "retarget (reproduces pre-grid seeding). Grid optimum "
                         "is 0.60 at spread 1.0.")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--state-passing", action="store_true",
                    help="carry KDA recurrent state across contiguous segments")
    ap.add_argument("--ramp-seq", action="store_true",
                    help="gradually increase sequence length during training")
    ap.add_argument("--gen", action="store_true", help="sample text at each eval")
    ap.add_argument("--max-seq", type=int, default=8192,
                    help="ramp target; with --ramp-seq the segment grows to this")
    ap.add_argument("--length-mix", action="store_true",
                    help="sample sequence length per step from LENGTH_MIX")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="trade compute for memory so longer segments fit")
    ap.add_argument("--build-cache", metavar="PATH",
                    help="one teacher pass: cache top-K logits, then exit")
    ap.add_argument("--logit-cache", metavar="PATH",
                    help="train from cached teacher top-K; no teacher forward")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--cache-tokens", type=int, default=1_000_000,
                    help="how many tokens to cache; guards against filling the "
                         "root filesystem, which can brick a Jetson")
    ap.add_argument("--taid", action="store_true",
                    help="TAID: KL to a time-interpolated student/teacher target")
    ap.add_argument("--lm-weight", type=float, default=0.0,
                    help="teacher-free CE term; distillation alone caps the "
                         "student at teacher quality")
    ap.add_argument("--mla-energy", type=float, default=None,
                    help="convert attention layers to latent KV, per-layer rank "
                         "by spectral energy, then recover")
    ap.add_argument("--resume", default=None,
                    help="continue from a resume file (weights + optimizer moments "
                         "+ LR schedule position + sampler RNG). Restarting from a "
                         "weight checkpoint alone is NOT the same run: it re-inits "
                         "Adam's moments and restarts OneCycleLR at step 0.")
    ap.add_argument("--mla-budget", type=int, default=None,
                    help="total latent rank summed over the 6 attention layers, "
                         "allocated non-uniformly by CARE water-filling on the "
                         "WHITENED spectra. 1536 is exactly 6x256, i.e. the same "
                         "4.00x KV as --mla-dc 256, but distributed to the layers "
                         "that need it: measured 5.02%% less activation error at "
                         "identical cache size. Requires --mla-covs.")
    ap.add_argument("--mla-dc", type=int, default=None,
                    help="fixed latent dim instead of adaptive")
    ap.add_argument("--mla-covs", default=None,
                    help="CARE whitening: path to saved per-layer input "
                         "covariances. Plain SVD minimizes ||W - W_hat||, which "
                         "is the wrong objective -- the model only cares about "
                         "||XW - XW_hat||. Whitening by X^T X optimizes "
                         "activation error instead. Measured here: 26.76pp of "
                         "ppl@2048 at d_c=256, 3.97pp at d_c=512.")
    ap.add_argument("--train-gate", action="store_true",
                    help="unfreeze linear_attn.in_proj_a, the DATA-DEPENDENT term "
                         "of the decay gate. A_log and dt_bias give a static "
                         "per-channel profile and were already trainable; "
                         "in_proj_a is the only term that makes the decay vary "
                         "per channel WITH CONTENT, and it is tiled from GDN's "
                         "per-head rows and frozen. Measured: the rank-32 gate "
                         "LoRA moved it by only 1.8% of its norm in 2.46M "
                         "tokens, so the diagonal never really forms. ~37.7M params.")
    ap.add_argument("--train-attn", action="store_true",
                    help="dense-train ALL linear_attn and self_attn parameters "
                         "(~273M). LoRA then only matters for the FFN. Memory is "
                         "fine (~1.5 GiB of 8-bit AdamW state); the risk is data "
                         "-- 273M params against a 999k-token cache is 273 "
                         "params/token.")
    ap.add_argument("--train-norms", action="store_true",
                    help="unfreeze every normalization gain. Costs 0.05M params "
                         "(0.006% of the model), so this is nearly free.")
    ap.add_argument("--allow-plain-svd", action="store_true",
                    help="permit an MLA conversion without covariances. Off by "
                         "default because plain SVD is known-worse here and the "
                         "failure is silent -- it just trains from a weaker "
                         "starting point and looks like a normal run.")
    ap.add_argument("--init-adapters", default=None,
                    help="start from previously trained adapters (merged into "
                         "the MLA factorization if converting)")
    ap.add_argument("--ckpt-above", type=int, default=8192,
                    help="only gradient-checkpoint windows LONGER than this. "
                         "Measured: checkpointing costs 1.39x at 2048 and 1.44x "
                         "at 8192, because it re-runs the forward and this model "
                         "is dispatch-bound (82%% of a step is CPU). It is still "
                         "required at 32768, where the un-checkpointed peak "
                         "extrapolates past 100 GiB from 26.84 GiB at 8192. "
                         "Set to 0 to checkpoint everything (the old behaviour).")
    ap.add_argument("--live-teacher", action="store_true",
                    help="run the teacher every step instead of reading a cache, "
                         "and use exact FULL-VOCABULARY KL. Removes the top-64 "
                         "truncation (which closes only ~5%% of the gap to full "
                         "distillation at this k), the cached-prefix alignment "
                         "constraint, the block-aligned sampling that collapsed "
                         "windows to ~91 distinct starts, and the disk budget. "
                         "Costs roughly +40-90%% wall clock.")
    ap.add_argument("--fullkl-chunk", type=int, default=512,
                    help="positions per full-vocab chunk; two C x V fp32 blocks "
                         "are live at once, so 512 is ~0.5 GiB each")
    ap.add_argument("--head-chunk", type=int, default=2048,
                    help="positions per lm_head chunk in the cached-KL path; "
                         "bounds peak head memory independently of seq length")
    ap.add_argument("--kda-rank", type=int, default=None,
                    help="override the LoRA rank on the KDA projections "
                         "(in_proj_qkv, in_proj_z, out_proj). Default 16.")
    ap.add_argument("--relora-every", type=int, default=0,
                    help="ReLoRA: merge every adapter into its base and restart "
                         "it from zero every N steps. Cumulative rank becomes "
                         "N_merges*r while only r is ever resident. 0 = off.")
    ap.add_argument("--lora-lr", type=float, default=None,
                    help="separate LR for LoRA adapters (A/B and the KDA gate "
                         "adapters). Without it they share the dense group's "
                         "rate, which is 5-10x below normal LoRA practice.")
    ap.add_argument("--layerscale", action="store_true",
                    help="per-channel (1+lam) scale on every residual branch "
                         "output, identity init (lam=0, bit-exact). Adds "
                         "full-rank row-scaling capacity that rank-r LoRA "
                         "cannot express -- the DoRA magnitude component. Only "
                         "non-redundant where the branch output is LoRA-only.")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    # DIFFERENT corpora: training on the eval set would make perplexity measure
    # memorization instead of recovery.
    train_ids = tok(open(TRAIN_DATA).read(), return_tensors="pt").input_ids[0]
    eval_ids = tok(open(EVAL_DATA).read(), return_tensors="pt").input_ids[0]
    print(f"train {len(train_ids):,} tok (fineweb-edu) | "
          f"eval {len(eval_ids):,} tok (wikitext, held out)", flush=True)
    print(f"dial={a.dial} seeded={a.seed_decay} steps={a.steps} seq={a.seq}",
          flush=True)
    print(f"epochs over train corpus: "
          f"{a.steps * a.seq / len(train_ids):.3f}", flush=True)

    # ---- teacher: the ORIGINAL model, untouched and frozen ----
    teacher = AutoModelForCausalLM.from_pretrained(
        ORIG, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if a.build_cache:
        n = build_cache(teacher, train_ids, a.build_cache, k=a.topk,
                        seq=a.seq, n_tokens=a.cache_tokens)
        print(f"cache built: {n:,} tokens -- rerun with --logit-cache {a.build_cache}")
        return 0

    cache = None
    if a.logit_cache:
        cache = load_cache(a.logit_cache)
        print(f"  logit cache: {cache['n_tokens']:,} tokens, k={cache['k']} "
              f"-- teacher forward SKIPPED during training", flush=True)

    # ---- student: converted, optionally seeded, dialled ----
    student = load_kda_model(CKPT, dtype=torch.bfloat16)
    if a.seed_decay:
        ta = a.seed_alpha if a.seed_alpha > 0 else None
        for l in get_trunk(student).layers:
            if hasattr(l, "linear_attn"):
                l.linear_attn.seed_decay_from_rope(target_alpha=ta)
        print(f"  decay seed: target median alpha "
              f"{ta if ta else '(none - pre-grid behaviour)'}", flush=True)
    keep, policy = {"nope": (0, "global"), "c1": (16, "local"),
                    "c0": (32, "local")}[a.dial]
    install_rope_dial(student, keep, policy)

    if a.state_passing:
        n_sp = enable_state_passing(student)
        print(f"  state passing enabled on {n_sp} KDA layers", flush=True)

    if a.grad_checkpoint:
        student.gradient_checkpointing_enable()
        student.config.use_cache = False
        print("  gradient checkpointing enabled", flush=True)

    # prior adapters load BEFORE conversion so their delta is merged into the
    # factorization rather than discarded by it
    if a.init_adapters:
        inject_lora(student, LORA_RULES, verbose=False)
        freeze_base(student)
        _sd = torch.load(a.init_adapters, map_location="cpu")
        student.load_state_dict({k: v.cuda() for k, v in _sd.items()}, strict=False)
        print(f"  init from {a.init_adapters.split('/')[-1]} "
              f"({len(_sd)} tensors)", flush=True)

    mla_latents = []
    if a.mla_energy is not None or a.mla_dc is not None or a.mla_budget is not None:
        covs = None
        if a.mla_covs:
            _c = torch.load(a.mla_covs, map_location="cpu")
            covs = {int(k): v.cuda().float() for k, v in _c.items()}
            print(f"  CARE whitening from {a.mla_covs.split('/')[-1]} "
                  f"({len(covs)} layer covariances)", flush=True)
        elif not a.allow_plain_svd:
            raise SystemExit(
                "refusing plain SVD for the MLA conversion. Measured on this "
                "model: plain SVD costs +35.16% ppl@2048 at d_c=256 where CARE "
                "whitening costs +8.40% -- a 26.76pp gap, and 3.97pp even at "
                "d_c=512. Build the covariances with src/save_covs.py and pass "
                "--mla-covs cache/kv_covs.pt, or --allow-plain-svd to override.")
        else:
            print("  plain SVD OVERRIDE: factorizing weight error, not "
                  "activation error -- known worse, you asked for it",
                  flush=True)
        info = convert_to_mla(student, d_c=a.mla_dc, budget=a.mla_budget,
                              energy=a.mla_energy if a.mla_energy else 0.95,
                              covs=covs)
        for l in get_trunk(student).layers:
            sa = getattr(l, "self_attn", None)
            if sa is not None and hasattr(sa.k_proj, "latent"):
                mla_latents.append(sa.k_proj.latent)

    rules = LORA_RULES
    if a.kda_rank:
        rules = [(p, a.kda_rank if p.startswith('linear_attn.') else r)
                 for p, r in LORA_RULES]
        print(f'  KDA projection LoRA rank -> {a.kda_rank}', flush=True)
    inject_lora(student, rules)
    # Substring matching against parameter names. LoRA-wrapped modules expose
    # their frozen base as "<mod>.base.weight", so a module-prefix pattern like
    # "linear_attn." unfreezes the dense base AND its adapter together -- which
    # is why no adapter merging is needed to switch a module from LoRA to dense.
    # after inject_lora, so lam scales (base + LoRA delta), not the frozen base
    if a.layerscale:
        install_layerscale(student)
    also = ("lora_A", "lora_B", "a_lora_A", "a_lora_B", "A_log", "dt_bias")
    if a.layerscale:
        also = also + ("ls_lambda",)
    if a.train_gate:
        # "diagonal dense, everything else adapted". The diagonal path is
        # in_proj_a -> A_log/dt_bias (A_log and dt_bias are already in `also`).
        # in_proj_b (16,1024) and conv1d (6144,1,4) are also dense, not because
        # they are diagonal but because LoRA is degenerate on them: rank 16 IS
        # full rank for in_proj_b and would cost more params than the weight,
        # and a depthwise conv has no cross-channel mixing to factorize (its
        # full rank is the kernel size, 4). 0.73 M params for both.
        also = also + ("in_proj_a.weight", "in_proj_b.weight", "conv1d.")
    if a.train_attn:
        also = also + ("linear_attn.", "self_attn.")
    if a.train_norms:
        also = also + ("norm",)
    n_tr = freeze_base(student, also_train=also)
    if a.train_gate or a.train_attn or a.train_norms:
        grp = {}
        for nm, p in student.named_parameters():
            if not p.requires_grad:
                continue
            if "linear_attn" in nm:   k = "KDA (linear_attn)"
            elif "self_attn" in nm:   k = "attention / MLA"
            elif "mlp." in nm:        k = "FFN (LoRA)"
            elif "norm" in nm:        k = "normalizers"
            else:                     k = "other"
            grp[k] = grp.get(k, 0) + p.numel()
        print("  trainable breakdown:", flush=True)
        for k in sorted(grp, key=lambda x: -grp[x]):
            print(f"    {k:<22}{grp[k]/1e6:>8.2f} M", flush=True)
    if mla_latents:
        # the latent down/up projections are small and are the thing that must
        # absorb the truncation, so train them densely rather than via LoRA
        extra = 0
        for lat in mla_latents:
            for prm in lat.parameters():
                prm.requires_grad_(True)
                extra += prm.numel()
        n_tr += extra
        print(f"  MLA latents trainable: {extra/1e6:.2f} M params "
              f"across {len(mla_latents)} layers", flush=True)
    # bf16 norm gains do not actually train. At w ~ 0.5 the bf16 ULP is 3.91e-3
    # while an Adam step at lr 2e-4 is ~2e-4, so the update rounds away and
    # q_norm/k_norm move by EXACTLY ZERO over the whole run; linear_attn.norm
    # and model.norm undershoot by 48-86x. Stage A's fold parks the two big
    # groups at exactly 0.0 -- the one operating point where bf16 survives --
    # so the three norms Stage A skipped are the three that break.
    # 55,552 params in fp32 is 222 KB, and both norm classes already promote
    # internally, so mixed dtype needs no module changes. Must happen BEFORE the
    # optimizer is constructed.
    n32 = 0
    for nm, p in student.named_parameters():
        if p.requires_grad and "norm" in nm and p.dtype != torch.float32:
            p.data = p.data.float()
            n32 += p.numel()
    if n32:
        print(f"  norm gains cast to fp32: {n32:,} params "
              f"(bf16 would round a 2e-4 step to zero)", flush=True)

    params = trainable_parameters(student)
    print(f"  trainable: {n_tr/1e6:.2f} M params in {len(params)} tensors", flush=True)

    # One group by default, which means LoRA adapters train at the DENSE-safe
    # rate. That is 5-10x below normal LoRA practice, and it silently handicaps
    # every LoRA-heavy arm: a surface ablation run this way measures the
    # learning rate, not the surface.
    if a.lora_lr:
        lora_keys = ("lora_A", "lora_B", "a_lora_A", "a_lora_B")
        lora_p, dense_p = [], []
        for nm, p in student.named_parameters():
            if not p.requires_grad:
                continue
            (lora_p if any(k in nm for k in lora_keys) else dense_p).append(p)
        groups = [{"params": dense_p, "lr": a.lr},
                  {"params": lora_p, "lr": a.lora_lr}]
        max_lr = [a.lr, a.lora_lr]
        print(f"  two parameter groups: {sum(p.numel() for p in dense_p)/1e6:.2f} M "
              f"dense @ {a.lr:g}, {sum(p.numel() for p in lora_p)/1e6:.2f} M "
              f"LoRA @ {a.lora_lr:g}", flush=True)
    else:
        groups, max_lr = params, a.lr

    opt = bnb.optim.AdamW8bit(groups, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=a.steps, pct_start=0.05)

    hist = {"loss": [], "eval": []}

    def evaluate(step):
        student.eval()
        was_sp = a.state_passing
        if was_sp:
            enable_state_passing(student, False)   # eval without carried state
        row = {"step": step}
        print(f"  [eval @ {step:>4}]", flush=True)
        for n in (2048, 8192):
            m = ce_and_topk(student, eval_ids, n, teacher=teacher)
            row[n] = m
            report(f"@{n}", m)
        if a.gen:
            for pr, out in zip(PROMPTS, sample(student, tok, PROMPTS, max_new=64)):
                print(f"    > {pr[:48]!r}\n      {out[:220]!r}", flush=True)
        hist["eval"].append(row)
        # Checkpoint at EVERY eval. The final-step weights are not reliably the
        # best ones: the mla-care-d256 run peaked at step 100 (+4.18% ppl@2048
        # against the uncompressed baseline) and then regressed past its own
        # untrained starting point (+9.30% by step 300), because 400 steps at
        # ~6.3k tok/step is ~2.5 epochs over a 999k-token logit cache. Saving
        # only at the end threw the good checkpoint away with no way to get it
        # back -- the eval curve survived and the weights did not.
        try:
            st = os.statvfs("/")
            if st.f_bavail * st.f_frsize > 8 * 2**30:      # keep 8 GiB headroom
                ck = (f"ckpt/"
                      f"adapters-{a.tag}-step{step}.pt")
                save_trainable(student, ck)
                # Overwritten each eval, so it stays ~1.1 GiB rather than growing.
                if step > 0 and _resume_ctx:
                    save_resume(f"ckpt/resume-{a.tag}.pt",
                                student, _resume_ctx["opt"], _resume_ctx["sched"],
                                step, _resume_ctx["gens"])
            else:
                print("    (skipping checkpoint: under 8 GiB free)", flush=True)
        except Exception as e:
            print(f"    (checkpoint save failed: {e})", flush=True)
        if was_sp:
            enable_state_passing(student, True)
        student.train()

    evaluate(0)
    t0 = time.perf_counter()
    seen = 0
    # mutable cell so the training loop can toggle checkpointing per window
    _ckpt_on = [bool(a.grad_checkpoint)]
    student.train()
    # Owned by main so their state can go into the resume file: a resumed run
    # that re-seeds these would draw the same windows it already trained on.
    _g = torch.Generator().manual_seed(1234)          # length mixture
    _bg = torch.Generator().manual_seed(0)            # window offsets
    # evaluate() closes over this. It is defined here, after the generators and
    # after opt/sched, but before any evaluate(step>0) call -- evaluate(0) runs
    # earlier and short-circuits on `step > 0`, so the name is never looked up
    # then. Closures resolve at call time, not definition time.
    _resume_ctx = {"opt": opt, "sched": sched, "gens": {"mix": _g, "batch": _bg}}
    if a.length_mix:
        global LENGTH_MIX
        # --live-teacher also has no cache, but it does NOT need the cap: its
        # full-vocab KL is chunked (_chunk_fullkl_terms), so neither model's
        # (T, 248320) logits are ever resident. This guard was written for the
        # old unchunked path and silently truncated the live mix to 8192-only
        # on its first run, which is exactly the kind of stale-guard failure the
        # save filter had.
        if cache is None and not a.live_teacher:
            capped = [(n, p) for n, p in LENGTH_MIX if n <= 8192]
            tot = sum(p for _, p in capped)
            LENGTH_MIX = [(n, p / tot) for n, p in capped]
            print("  length mix CAPPED at 8192: the full-vocab KL needs teacher "
                  "AND student logits resident, ~15.2 GiB each at 32k, plus a "
                  "gradient buffer. --logit-cache removes the teacher's copy "
                  "and switches to the chunked top-K head, which removes the "
                  "student's too -- only then are the long draws affordable.",
                  flush=True)
        sl = lambda st: sample_length(_g)
        print(f"  length mix: {LENGTH_MIX}", flush=True)
        print(f"  expected tokens/step {mix_expected_tokens():.0f} "
              f"(vs {max(n for n,_ in LENGTH_MIX)} if always long -- "
              f"{max(n for n,_ in LENGTH_MIX)/mix_expected_tokens():.1f}x cheaper)",
              flush=True)
    else:
        sl = lambda st: seq_len_at(st, a.steps, a.seq, a.max_seq, a.ramp_seq)
        print(f"  segment length: {sl(0)} -> {sl(a.steps)}"
              f"{' (ramped)' if a.ramp_seq else ''}", flush=True)
    # The cache is keyed by position in train_ids, so sampling must stay inside
    # the cached prefix -- otherwise offsets run past the end and silently read
    # the wrong teacher distribution (or go out of bounds).
    sample_ids = train_ids
    if cache is not None:
        sample_ids = train_ids[:cache["n_tokens"]]
        print(f"  sampling restricted to the cached prefix "
              f"({cache['n_tokens']:,} of {len(train_ids):,} tokens)", flush=True)
        # The CACHE is the real data budget, not the corpus. "epochs over train
        # corpus" printed at startup reads 0.03 and looks safe, while the run is
        # actually cycling a 1M-token slice 2.5 times. That discrepancy is what
        # made mla-care-d256 peak at step 100 and then decay.
        per_step = mix_expected_tokens() if a.length_mix else a.seq
        ep = a.steps * per_step / max(cache["n_tokens"], 1)
        print(f"  epochs over the CACHE: {ep:.2f}  "
              f"({a.steps} steps x {per_step:.0f} tok/step)", flush=True)
        if ep > 1.0:
            print(f"  WARNING: {ep:.2f} epochs over the cached prefix -- expect "
                  f"it to fit the slice and lose held-out quality. Measured "
                  f"peak is near 1 epoch (~{int(cache['n_tokens'] / per_step)} "
                  f"steps here). Either build a bigger cache (--build-cache, "
                  f"384 B/token, so 5M tokens is 1.9 GB) or cut --steps.",
                  flush=True)
        if max(sl(st) for st in (0, a.steps // 2, a.steps)) > cache["n_tokens"]:
            raise SystemExit("cache shorter than the sequence length")
    # Align sampling to the teacher's cache blocks, and refuse to train on a
    # window longer than the teacher ever saw. Before this, 84.6% of training
    # tokens came from windows longer than the cache's 2048-token block, so the
    # teacher supervising a 32k window had seen ~1024 tokens of it.
    align = int(cache["seq"]) if cache is not None else 1
    if cache is not None:
        longest = max(sl(st) for st in (0, a.steps // 2, a.steps))
        if longest > align:
            raise SystemExit(
                f"length mix draws up to {longest} but the cache was built at "
                f"seq={align}. The teacher never saw more than {align} tokens, "
                f"so forward KL would push the student flatter than the context "
                f"allows -- penalising exactly the retrieval we are trying to "
                f"recover. Rebuild with --build-cache at the longer seq.")
        print(f"  cache blocks: seq={align}, offsets aligned "
              f"(teacher and student see the same prefix)", flush=True)
    start_step = 0
    if a.resume:
        _rs = torch.load(a.resume, map_location="cpu")
        student.load_state_dict({k: v.cuda() for k, v in _rs["weights"].items()},
                                strict=False)
        opt.load_state_dict(_rs["opt"])
        # OneCycleLR's state carries total_steps from the ORIGINAL run, so
        # restoring it silently overrides --steps and the schedule then refuses
        # to advance past the old total ("Tried to step N+1 times"). Resuming
        # with a different --steps is not continuing a run -- it is a different
        # LR trajectory -- so refuse it here with the reason rather than failing
        # one step past the old horizon.
        _prev_total = _rs["sched"].get("total_steps")
        if _prev_total is not None and int(_prev_total) != int(a.steps):
            raise SystemExit(
                f"--resume was written by a run with --steps {_prev_total}, but "
                f"this invocation passes --steps {a.steps}. OneCycleLR's shape "
                f"depends on total_steps, so continuing with a different value "
                f"is a different schedule, not a resume. Re-run with "
                f"--steps {_prev_total}, or start fresh without --resume.")
        sched.load_state_dict(_rs["sched"])
        _g.set_state(_rs["gens"]["mix"])
        _bg.set_state(_rs["gens"]["batch"])
        start_step = int(_rs["step"])
        print(f"  RESUMED from {a.resume.split('/')[-1]} at step {start_step}/{a.steps} "
              f"(optimizer moments, LR position and sampler RNG restored)", flush=True)
        del _rs
    gen = (sequential_batches(sample_ids, sl, a.steps - start_step) if a.state_passing
           else batches(sample_ids, sl, a.steps - start_step, align=align, gen=_bg))
    for step, (batch, new_doc, batch_offset) in enumerate(gen, start=start_step + 1):
        if a.state_passing and new_doc:
            reset_state(student)
        x = batch.cuda()
        seen += x.numel()
        # Length-conditional checkpointing. Toggling is a flag flip on the
        # modules, so it costs nothing per step, and it buys ~1.4x on every
        # window short enough to hold its own activations.
        if a.grad_checkpoint:
            want = x.shape[1] > a.ckpt_above
            if want != _ckpt_on[0]:
                (student.gradient_checkpointing_enable() if want
                 else student.gradient_checkpointing_disable())
                student.config.use_cache = False
                _ckpt_on[0] = want
        L = x.shape[1]
        s_logits = t_logits = None
        if a.live_teacher:
            # Exact full-vocabulary KL against a teacher conditioned on the
            # student's actual prefix. No cache, so no support truncation, no
            # block alignment, and windows may start anywhere in the corpus.
            with torch.no_grad():
                h_t = get_trunk(teacher)(input_ids=x).last_hidden_state[0]
            h_s = get_trunk(student)(input_ids=x).last_hidden_state[0]
            W_lm = student.lm_head.weight
            tot, n = 0.0, 0
            for i in range(0, L, a.fullkl_chunk):
                j = min(i + a.fullkl_chunk, L)
                terms = checkpoint(_chunk_fullkl_terms, h_s[i:j], h_t[i:j],
                                   W_lm, use_reentrant=False)
                tot = tot + terms.sum()
                n += j - i
            loss = tot / max(n, 1)
            del h_t
        elif cache is not None:
            # cached path: no teacher forward, and KL restricted to the
            # teacher's top-K support instead of a 248,320-way softmax.
            # The logits are never materialized -- see _chunk_kl_terms.
            off = batch_offset
            assert off + L <= cache["n_tokens"], (
                f"cache miss: offset {off}+{L} exceeds {cache['n_tokens']}")
            tv = cache["vals"][off:off + L].cuda()
            ti = cache["idxs"][off:off + L].cuda()
            lam = step / max(1, a.steps)
            h = get_trunk(student)(input_ids=x).last_hidden_state[0]
            W_lm = student.lm_head.weight
            tot, n = 0.0, 0
            for i in range(0, L, a.head_chunk):
                j = min(i + a.head_chunk, L)
                terms = checkpoint(_chunk_kl_terms, h[i:j], W_lm, tv[i:j],
                                   ti[i:j], lam, a.taid, use_reentrant=False)
                tot = tot + terms.sum()
                n += j - i
            loss = tot / max(n, 1)
            if a.lm_weight > 0:
                ce_tot, ce_n = 0.0, 0
                for i in range(0, L - 1, a.head_chunk):
                    j = min(i + a.head_chunk, L - 1)
                    ce = checkpoint(_chunk_ce_terms, h[i:j], W_lm,
                                    x[0, i + 1:j + 1], use_reentrant=False)
                    ce_tot = ce_tot + ce.sum()
                    ce_n += j - i
                loss = loss + a.lm_weight * (ce_tot / max(ce_n, 1))
        else:
            # full-vocab path: both sets of logits must be resident, so the
            # length mix is capped above. No gather trick applies here.
            s_logits = student(input_ids=x).logits[0]
            with torch.no_grad():
                t_logits = teacher(input_ids=x).logits[0]
            loss = kl_loss(s_logits, t_logits)
            if a.lm_weight > 0:
                loss = loss + a.lm_weight * F.cross_entropy(
                    s_logits[:-1].float(), x[0, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if a.relora_every and step > 0 and step % a.relora_every == 0:
            nm = merge_and_restart(student, opt)
            print(f'  [relora] merged and restarted {nm} adapters at step '
                  f'{step} (merge {step//a.relora_every})', flush=True)
        if a.state_passing:
            promote_state(student)   # after backward, so recompute stays valid
        hist["loss"].append(loss.item())
        del t_logits, s_logits
        if step % 25 == 0:
            el = time.perf_counter() - t0
            tok_s = seen / el
            print(f"  step {step:>4}/{a.steps}  KL {loss.item():8.4f}  "
                  f"{tok_s:6.1f} tok/s  {el/60:5.1f} min", flush=True)
        if step % a.eval_every == 0:
            evaluate(step)
        torch.cuda.empty_cache()

    evaluate(a.steps)
    out = f"logs/recovery-{a.tag}.json"
    json.dump({"args": vars(a), **hist}, open(out, "w"), indent=2)
    # save the trained parameters -- without this a run's weights are lost and
    # only the eval curve survives.
    adp = f"ckpt/adapters-{a.tag}.pt"
    n_saved = save_trainable(student, adp)
    print(f"  saved {n_saved} trainable tensors", flush=True)
    print(f"\nwrote {out}\nwrote {adp}")

    e0, e1 = hist["eval"][0], hist["eval"][-1]
    print("\n=== RECOVERY ===")
    for n in (2048, 8192):
        a0, a1 = e0[n], e1[n]
        print(f"  @{n}: CE {a0['ce']:.4f} -> {a1['ce']:.4f} | "
              f"ppl {a0['ppl']:.3f} -> {a1['ppl']:.3f} "
              f"({(a1['ppl']-a0['ppl'])/a0['ppl']*100:+.2f}%) | "
              f"top1 {a0['top1']:.2f} -> {a1['top1']:.2f}%")
    print(f"  tokens seen: {seen / 1e6:.2f} M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
