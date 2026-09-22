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
import sys, os, json, time, argparse, random, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius.models.kda import (load_kda_model, enable_state_passing, reset_state,
                       promote_state, fuse_gate_lora)
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import (inject_lora, freeze_base, trainable_parameters, merge_vera,
                  merge_and_restart, merged_base_names, resize_lora,
                  rules_from_checkpoint, inject_vera, unwrap_lora)
from mercurius.eval.characterize import perplexity
from mercurius.eval.suite import ce_and_topk, sample, report, PROMPTS
import bitsandbytes as bnb
from mercurius.recovery.logit_cache import build_cache, load_cache, topk_kl, taid_kl
from mercurius.surgery.transmla import convert_to_mla
from mercurius.adapters.layerscale import install_layerscale
from mercurius.paths import (BASE_MODEL, CKPT_DIR, FINEWEB, LOGS_DIR, STAGE_AB,
                             TEACHER_MODEL, WIKITEXT)

CKPT = str(STAGE_AB)
ORIG = str(BASE_MODEL)
TEACHER = str(TEACHER_MODEL)
TRAIN_DATA = str(FINEWEB)   # train
EVAL_DATA  = str(WIKITEXT)      # eval, held out

LORA_RULES = [
    ("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
    ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),   # the dialled layers
    # KDA projections carry a per-rule alpha. Under rsLoRA the scale is
    # alpha/sqrt(r), so alpha=4 gives 1.0 at rank 16, 0.707 at 32, 0.5 at 64 --
    # the update magnitude stays put as rank rises instead of growing like
    # sqrt(r), which is what the classic convention does and what made the
    # earlier rank-64 arm uninterpretable. The attention and FFN groups have no
    # per-rule alpha, so they stay on the classic convention at scale 1.0 and a
    # KDA rank sweep cannot move them.
    ("linear_attn.out_proj", 16, 4.0), ("linear_attn.in_proj_qkv", 16, 4.0),
    # in_proj_z is 16.5% of KDA and had no adaptation path at all: frozen under
    # every LoRA-only arm, dense only under --train-attn. Under --train-attn the
    # base unfreezes alongside this adapter, so dense runs gain 0.88 M redundant
    # adapter params -- val-full's recorded numbers predate this rule.
    ("linear_attn.in_proj_z", 16, 4.0),
    # FFN at 32, not 16. The rank-16 adapter was saturated: effective rank
    # 12.0-12.3 of 16 with sigma1/sigma16 = 3.8, against ~1.3 for a random
    # init, so training had shaped the spectrum and still used nearly every
    # direction. Separately, the double-adapter bug gave these targets two
    # parallel rank-16 adapters and removing it cost 0.6% (val-full 17.292 vs
    # valfix 17.397 from the same start) -- the accident was supplying the
    # capacity the measurement asked for. One adapter at 32 is the same
    # capacity without the duplicate.
    ("mlp.gate_proj", 32), ("mlp.up_proj", 32), ("mlp.down_proj", 32),
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


import math

LOG2 = math.log(2.0)
CE_EPS = 1e-4

from torch.utils.checkpoint import checkpoint


def save_trainable(model, path, save_bases=False):
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
    # A merged base is frozen but CHANGED, so requires_grad cannot see it -- but
    # storing it is the wrong fix. It is 0.91 GB of a 0.92 GB checkpoint, 99% of
    # the file, and it is DETERMINISTIC: stage-AB plus the init delta, replayed
    # identically by build(). A checkpoint written without it rebuilt to
    # +0.0000% (adapters-allvera-ce-best.pt, 0.01 GB). Saving it made every
    # checkpoint 92x larger for nothing.
    #
    # The cost is a dependency: ckpt/qwen3.5-0.8b-stageAB and
    # ckpt/adapters-combined.pt become load-bearing for every checkpoint that
    # omits the bases. Pass --save-merged-bases to make a self-contained copy.
    if save_bases:
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


def taid_space_for(mode):
    """The TAID interpolation space that is NOT degenerate for a divergence.

    Derivation (student logits z_s, teacher log-probs log p_T, target m built
    from a DETACHED copy of the student, gradients w.r.t. z_s):

      logit space  m = softmax((1-t) z_s + t z_T)      (TAID, arXiv:2501.16937)
        forward  KL(m||s):  grad = s - m               genuine; ~ t*grad KL(s||T)
                                                       at small t, grad KL(T||s)
                                                       at t=1 (reverse -> forward)
        reverse  KL(s||m):  log s - log m = t (log s - log p_T) + const, so
                            grad = t * grad KL(s||T)   EXACTLY: a loss scale

      prob space   m = (1-t) s + t p_T
        forward  KL(m||s):  grad = s - m = t (s - p_T) = t * grad KL(T||s)
                                                       EXACTLY: a loss scale
        reverse  KL(s||m):  genuine. This is DistiLLM's skew reverse KL
                            (Ko et al., ICML 2024) with alpha = 1 - t; at small
                            t its gradient is ~ t * grad KL(T||s), so the
                            curriculum runs forward -> reverse.

    So in each space one of the two divergences turns TAID into nothing but a
    ramp on the loss scale, i.e. a second LR warmup. The previous code used
    prob space for every mode, which made forward+TAID exactly that; the
    paper's logit space makes reverse+TAID exactly that. Checked numerically in
    experiments/test_objective.py.
    """
    return {"forward": "logit", "reverse": "prob"}.get(mode)


def _chunk_div_terms(h_s_c, h_t_c, W_s, W_t, mode="forward", lam=1.0,
                     tgt_c=None, ce_beta=1.0, ce_mix=0.0, taid_space="prob",
                     rev_w=1.0):
    """Per-position (divergence, data) terms, full vocabulary, in nats.

    Returns a (2, C) tensor: row 0 the distillation divergence, row 1 the data
    term (zeros when ce_beta == 0). The loss is row0 + ce_beta * row1. They are
    returned separately because TAID's adaptive schedule reads the distillation
    term alone.

    W_s and W_t are the two models' OWN output heads. They used to be one
    matrix, because the teacher was the student's unmodified original and the
    two shared a tied embedding. A different-size teacher has a different
    hidden width and its own head, and projecting its hidden states through the
    student's head is not merely wrong, it does not type-check.

    Both distributions are available explicitly here, so every mode is a direct
    expression -- no sampling, no estimator, same cost as forward KL. Reverse KL
    normally needs samples from the student (MiniLLM uses policy gradients for
    exactly this); it is cheap only because the live teacher gives full logits.

      forward  KL(t || s)   mode-covering
      reverse  KL(s || t)   mode-seeking: the principled choice when the student
                            is structurally incapable of matching the teacher,
                            which here is both architectural (linear attention,
                            compressed KV, NoPE) and a 4B-vs-27B capacity gap
      js                    symmetric, bounded
      TAID                  target built from the student's own (detached)
                            distribution and the teacher's, moving to the
                            teacher as lam -> 1. See taid_space_for for which
                            interpolation space is meaningful under which mode.

    THE DATA TERM, in excess nats:

        CE(y, student) - CE(y, teacher) = -log p_s(y) + log p_t(y)

    Zero when the student equals the teacher, negative when it beats it on the
    true token, and on the distillation term's scale by construction, so
    beta = 1 is meaningful. The offset log p_t(y) carries no gradient; it only
    makes the logged loss read as excess.

    This is CE in every mode, NOT D(y || s) - D(y || t) in the chosen
    divergence, which is what it used to be. CE is exactly forward KL against a
    one-hot, so in forward mode the two agree (up to the eps smoothing the old
    form needed). In reverse mode D(y || s) = KL(s || y_eps) =
    -H(s) + (1 - s_y) log(V/eps) + ..., minimised only by a point mass with a
    per-nat weight of log(V/eps) ~ 21.6 -- measured on the 0.8B as ppl@8192
    20.377 -> 104.972 in 100 steps. The excess-nats idea was right; carrying it
    through the reverse divergence was the error, because "cross-entropy in
    reverse" is not a cross-entropy.

    ce_mix (q = (1-w) p_teacher + w delta_y) is kept as an alternative that
    changes the target instead of adding a term.
    """
    lg_t = (h_t_c @ W_t.T).float()
    t_lp = F.log_softmax(lg_t, -1)
    del lg_t
    lg_s = (h_s_c @ W_s.T).float()
    s_lp = F.log_softmax(lg_s, -1)
    del lg_s
    return _terms_from_logprobs(s_lp, t_lp, tgt_c, mode, lam, ce_beta, ce_mix,
                                taid_space, rev_w)


def _terms_from_logprobs(s_lp, t_lp, tgt_c, mode="forward", lam=1.0,
                         ce_beta=1.0, ce_mix=0.0, taid_space="prob", rev_w=1.0):
    """The (divergence, data) terms of _chunk_div_terms, from log-probs.

    Split out so the batched MTP path (_chunk_mtp_terms) computes exactly the
    same per-row objective from logits it formed more cheaply."""
    data = torch.zeros(s_lp.shape[0], device=s_lp.device, dtype=s_lp.dtype)
    if ce_beta != 0.0 and tgt_c is not None:
        idx = tgt_c.view(-1, 1).long()
        # excess over the UNMODIFIED teacher, before any mix or TAID
        data = t_lp.gather(-1, idx).squeeze(-1) - s_lp.gather(-1, idx).squeeze(-1)
    if ce_mix > 0.0 and tgt_c is not None:
        V = t_lp.shape[-1]
        y = torch.full_like(t_lp, CE_EPS / V)
        y.scatter_(-1, tgt_c.view(-1, 1).long(), 1.0 - CE_EPS + CE_EPS / V)
        q = (1.0 - ce_mix) * t_lp.exp() + ce_mix * y
        t_lp = (q / q.sum(-1, keepdim=True).clamp_min(1e-9)).clamp_min(1e-9).log()
    # TAID walks from the student's own distribution toward the target (after
    # the mix, since the target is the thing being approached). lam = 1 leaves
    # t_lp untouched, so every mode is bit-identical to plain distillation.
    if lam < 1.0:
        if taid_space == "logit":
            t_lp = F.log_softmax((1.0 - lam) * s_lp.detach() + lam * t_lp, -1)
        else:
            q0 = (1.0 - lam) * s_lp.exp().detach() + lam * t_lp.exp()
            t_lp = (q0 / q0.sum(-1, keepdim=True).clamp_min(1e-9)).clamp_min(1e-9).log()

    def D(a_lp, b_lp):
        """Divergence from target a to model b, in nats, in the chosen mode."""
        if mode == "forward":
            return (a_lp.exp() * (a_lp - b_lp)).sum(-1)
        if mode == "reverse":
            return (b_lp.exp() * (b_lp - a_lp)).sum(-1)
        if mode == "js":
            m = torch.logaddexp(a_lp, b_lp) - LOG2
            return 0.5 * ((a_lp.exp() * (a_lp - m)).sum(-1)
                          + (b_lp.exp() * (b_lp - m)).sum(-1))
        if mode == "jeffreys":
            # KL(target||model) + w * KL(model||target): the symmetric KL, and
            # the direct response to a trade measured twice on this model.
            # Forward alone (mode-covering) gave the best perplexity and the
            # worst multi-item retrieval; reverse plus TAID (mode-seeking) gave
            # the worst perplexity and the best retrieval -- EM 78.3%, NLL 0.213,
            # better than any forward arm. A peaked retrieval distribution wants
            # mode-seeking; language modelling wants mode-covering. Carrying both
            # terms asks for both instead of choosing.
            #
            # Not the same as "js": Jensen-Shannon measures each side against the
            # MIDPOINT, which bounds the loss at log 2 and softens both
            # pressures. Jeffreys keeps each direction at full strength. js is
            # also implemented and has never been run.
            return ((a_lp.exp() * (a_lp - b_lp)).sum(-1)
                    + rev_w * (b_lp.exp() * (b_lp - a_lp)).sum(-1))
        raise ValueError(mode)

    return torch.stack([D(t_lp, s_lp), data])


class TAIDSchedule:
    """TAID's adaptive interpolation coefficient, as in the reference code
    (SakanaAI/TAID, distil_losses/taid.py):

        delta  = (J_prev - J) / J_prev          relative drop in the distil loss
        m      = beta m + (1 - beta) delta
        t_next = min(t_end, max(t_linear(step), t + alpha sigmoid(m) (1 - t)))

    t_linear ramps t_start -> t_end over the run, so the schedule is never
    slower than linear and speeds up while the loss is falling. Defaults are the
    reference's: t_start 0.4, t_end 1.0, alpha 5e-4, beta 0.99.

    The previous implementation was a linear ramp from 0: at step 1 the target
    was 1/steps of the way to the teacher, which in prob space is ~0 signal, and
    it had no adaptive term at all.
    """

    def __init__(self, total, t_start=0.4, t_end=1.0, alpha=5e-4, beta=0.99,
                 adaptive=True):
        self.total, self.t_start, self.t_end = total, t_start, t_end
        self.alpha, self.beta, self.adaptive = alpha, beta, adaptive
        self.t, self.m, self.prev = t_start, 0.0, None

    def update(self, step, J):
        lin = self.t_start + (self.t_end - self.t_start) * min(1.0, step / max(self.total, 1))
        if self.adaptive and self.prev is not None:
            delta = (self.prev - J) / (self.prev + 1e-15)
            self.m = self.beta * self.m + (1.0 - self.beta) * delta
            step_t = self.alpha * (1.0 / (1.0 + math.exp(-self.m))) * (1.0 - self.t)
            self.t = min(self.t_end, max(lin, self.t + step_t))
        else:
            self.t = min(self.t_end, max(lin, self.t))
        self.prev = J
        return self.t

    def state_dict(self):
        return {"t": self.t, "m": self.m, "prev": self.prev}

    def load_state_dict(self, d):
        self.t, self.m, self.prev = d["t"], d["m"], d["prev"]


def _chunk_mtp_terms(h_s_c, z_c, h_t_ext, W_s, W_t, tgt_ext, n_rows,
                     mode="forward", lam=1.0, ce_beta=1.0, ce_mix=0.0,
                     taid_space="prob", rev_w=1.0):
    """Main loss AND every MTP head for one chunk, sharing the vocab work.

    h_s_c    (C, d_s)      student states at positions p = i .. i+C-1
    z_c      (C, K, d_s)   MTP head outputs at the same positions
    h_t_ext  (C+K', d_t)   teacher states at p = i .. i+C+K'-1 (K' <= K, cut at
                           the sequence end)
    tgt_ext  (C+K',)       tokens at p+1: row r predicts tgt_ext[r]
    n_rows   (K+1,) ints   valid rows per offset j (the end of the sequence
                           removes targets for the far heads)

    Offset j (0 = the trunk's t+1, j >= 1 = head j predicting t+1+j) at
    position p is distilled against the TEACHER'S row p+j: its own next-token
    prediction for the same token. So one teacher log-softmax over C+K' rows
    serves all K+1 offsets, instead of recomputing (K+1) C rows, and the
    student's (K+1) C rows go through ONE GEMM against W_s. The per-row
    objective is _terms_from_logprobs, i.e. exactly what _chunk_div_terms
    computes; test_mtp_batched.py checks the two agree.

    Returns (2, K+1, C): [divergence, data] per offset and row, zero where a
    row has no target.
    """
    C, K, d = z_c.shape
    t_lp = F.log_softmax((h_t_ext @ W_t.T).float(), -1)
    S = torch.cat([h_s_c.unsqueeze(1), z_c], dim=1).reshape(C * (K + 1), d)
    s_lp = F.log_softmax((S @ W_s.T).float(), -1).view(C, K + 1, -1)
    out = torch.zeros(2, K + 1, C, device=s_lp.device, dtype=s_lp.dtype)
    for j in range(K + 1):
        n = int(n_rows[j])
        if n <= 0:
            continue
        out[:, j, :n] = _terms_from_logprobs(
            s_lp[:n, j], t_lp[j:j + n], tgt_ext[j:j + n], mode, lam, ce_beta,
            ce_mix, taid_space, rev_w)
    return out


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


def save_resume(path, model, opt, sched, step, gens, save_bases=False, taid=None):
    """Everything needed to continue a run exactly where it stopped.

    Weight checkpoints alone are NOT resumable: restarting from them re-inits the
    optimizer moments and restarts OneCycleLR from step 0, so the run continues on
    a different trajectory. This session lost ~100 minutes of a 300-step run twice
    because a mid-flight environment change (a newly built kernel, a stale guard)
    could only be adopted by starting over.

    Written once per eval and OVERWRITTEN, so it costs a constant ~1.1 GiB rather
    than growing with the number of evals.
    """
    # requires_grad alone is not the trained state: a base that absorbed a merge
    # is frozen and changed, so resuming from a requires_grad-only snapshot
    # restarts from unmerged weights -- a different model, silently.
    _keep = {n for n, p in model.named_parameters() if p.requires_grad}
    _sd = model.state_dict()
    if save_bases:
        # Off by default for the same reason as save_trainable: the bases are a
        # deterministic replay of stage-AB plus the init delta, and storing them
        # is what made a resume file 0.94 GB instead of ~30 MB. A resumed run
        # rebuilds the structure before loading this, so the replay has already
        # happened by the time these weights land.
        _keep |= {n for n in merged_base_names(model) if n in _sd}
    torch.save({"step": step,
                "weights": {n: _sd[n].detach().cpu() for n in sorted(_keep)},
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "gens": {k: g.get_state() for k, g in gens.items()},
                "taid": taid.state_dict() if taid is not None else None}, path)


def batches(ids, seq_len_fn, steps, seed=0, align=1, gen=None, spans=None):
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
        if spans is not None:
            # pick a document long enough, then a window INSIDE it, so the
            # window never straddles a boundary
            ok = [(s, e) for s, e in spans if e - s > seq_len]
            if not ok:
                raise SystemExit(f"no document reaches {seq_len} tokens")
            s, e = ok[int(torch.randint(0, len(ok), (1,), generator=g))]
            i = s + int(torch.randint(0, e - s - seq_len, (1,), generator=g))
        else:
            hi = len(ids) - seq_len - 1
            i = int(torch.randint(0, hi, (1,), generator=g))
        if align > 1:
            i = (i // align) * align
        yield ids[i:i + seq_len].unsqueeze(0), False, i


def _docs_and_spans(tok, path, min_len):
    """Per-document token tensors plus which of them qualify for sampling."""
    docs = [d for d in open(path).read().split("\n\n") if d.strip()]
    out = []
    for d in docs:
        ids = tok(d, return_tensors="pt", add_special_tokens=False).input_ids[0]
        out.append(ids)
    return out


def interleave_corpora(a_docs, b_docs, min_len):
    """Distribute b's documents evenly through a's, then build stream + spans.

    Appending instead of interleaving is silently fatal under --state-passing:
    sequential_batches starts at position 0 and walks forward, so 150 steps at
    8192 cover the first 1.23 M tokens of a 37.45 M stream and NEVER reach
    documents appended at the end. Measured: every on-policy step reported "no
    restatement header" because no window had come from a synthetic document --
    the mix was 0%, not the 19.9% the log claimed.
    """
    if not b_docs:
        merged = a_docs
    else:
        every = max(1, len(a_docs) // len(b_docs))
        merged, bi = [], 0
        for i, d in enumerate(a_docs):
            merged.append(d)
            if i % every == every - 1 and bi < len(b_docs):
                merged.append(b_docs[bi]); bi += 1
        merged.extend(b_docs[bi:])
    chunks, spans, pos = [], [], 0
    for ids in merged:
        n = int(ids.numel())
        if n >= min_len:
            spans.append((pos, pos + n))
        chunks.append(ids)
        pos += n
    import torch as _t
    return _t.cat(chunks), spans


def tokenize_by_document(tok, path, min_len):
    """Tokenize per document and return the stream plus per-document spans.

    Without this the corpus is one concatenated stream and a window is whatever
    happens to be adjacent. Measured on fineweb_edu.txt: median document 550
    tokens, so an 8192 window spans about 15 unrelated documents and contains no
    dependency longer than a couple of thousand tokens. Training a long-context
    repair on that cannot work, whatever the architecture does.

    Returns spans only for documents that are at least min_len long, so a window
    drawn inside one is entirely within a single document.
    """
    docs = [d for d in open(path).read().split("\n\n") if d.strip()]
    chunks, spans, pos = [], [], 0
    for d in docs:
        ids = tok(d, return_tensors="pt", add_special_tokens=False).input_ids[0]
        n = int(ids.numel())
        if n >= min_len:
            spans.append((pos, pos + n))
        chunks.append(ids)
        pos += n
    stream = torch.cat(chunks)
    kept = sum(e - s for s, e in spans)
    print(f"  document-aware: {len(docs):,} docs, {len(spans):,} at least "
          f"{min_len} tokens ({kept/1e6:.1f} M of {pos/1e6:.1f} M tokens usable)",
          flush=True)
    if not spans:
        raise SystemExit(
            f"no document in {path.split('/')[-1]} reaches {min_len} tokens. "
            f"Use src/get_long_data.py to build a long-document corpus, or "
            f"lower --seq.")
    return stream, spans


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
    ap.add_argument("--on-policy", type=float, default=0.0, metavar="FRAC",
                    help="fraction of steps that distil on the STUDENT's own "
                         "generations instead of corpus text (GKD). Fixes exposure "
                         "bias: teacher-forced training never asks the model to "
                         "recover from its own mistake, which is what emitting "
                         "several retrieved values in sequence requires. Reported "
                         "to turn quadratic error accumulation into linear")
    ap.add_argument("--on-policy-gen", type=int, default=256, metavar="N",
                    help="tokens the student generates on an on-policy step. The "
                         "prompt is the window minus N, so context length is "
                         "unchanged and only N decode steps are added")
    ap.add_argument("--rev-weight", type=float, default=1.0, metavar="W",
                    help="weight on the reverse term of --divergence jeffreys")
    ap.add_argument("--ce-mix", type=float, default=0.0, metavar="W",
                    # DEPRECATED. A mixing weight in [0,1] is exactly the
                    # arbitrary coefficient the CE-excess formulation exists to
                    # avoid. Kept at 0 only so old command lines still parse.
                    help="mix the ground truth INTO the distillation target with "
                         "weight W in [0,1]: q = (1-W)*p_teacher + W*onehot. "
                         "Unlike --ce-beta this carries the teacher term's own "
                         "scale, so it is portable across divergences; --ce-beta "
                         "in reverse mode is ~216x data-dominated and diverges")
    ap.add_argument("--synth-data", default=None, metavar="PATH",
                    help="extra corpus of synthetic multi-item recall documents "
                         "(src/synth_recall.py). Appended to the document pool, so "
                         "the mix fraction is its share of SPANS")
    ap.add_argument("--save-merged-bases", action="store_true",
                    help="also store the merged base weights in the checkpoint. "
                         "They are 99%% of the file and are replayable from "
                         "stage-AB plus --init-adapters, so this is only for a "
                         "self-contained copy that survives losing those files")
    ap.add_argument("--phq-lr", type=float, default=1e-3,
                    help="learning rate for the per-head query maps. Separate "
                         "from --lr and --vera-lr: R is identity-initialised, so "
                         "3e-5 leaves it 0.42%% off identity after 150 steps and "
                         "1e-2 can erase the identity entirely")
    ap.add_argument("--per-head-q", action="store_true",
                    help="give every query head its own effective key via a "
                         "per-head map on q (equivalent to widening up_k). "
                         "Identity at init; 3.15 M params, cache and FLOPs unchanged")
    ap.add_argument("--gdn2", action="store_true",
                    help="lift the KDA layers to Gated DeltaNet-2 (arXiv:2605.22791): "
                         "the scalar write gate becomes a channel-wise erase gate "
                         "on keys and write gate on values. Exact at init.")
    ap.add_argument("--decay-phase", type=float, default=None, metavar="C",
                    help="enable the decay-tied rotary phase on the linear-attention "
                         "layers with this init for c (0 is the identity)")
    ap.add_argument("--resume-every", type=int, default=25, metavar="N",
                    help="write the resume file every N steps, independent "
                         "of --eval-every (0 disables)")
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
    ap.add_argument("--no-taid", dest="taid", action="store_false",
                    help="disable the TAID target schedule (on by default)")
    ap.set_defaults(taid=True)
    ap.add_argument("--ce-beta", type=float, default=1.0,
                    help="weight on the data term, which is expressed as EXCESS "
                         "nats over the teacher: D(y||student) - D(y||teacher), "
                         "in whatever divergence --divergence selects. Zero when "
                         "the student equals the teacher, negative when it beats "
                         "it, so beta=1 is principled rather than a tuned "
                         "constant. 0 gives pure distillation, which caps the "
                         "student at teacher quality.")
    ap.add_argument("--divergence", default="js",
                    choices=["forward", "reverse", "js", "jeffreys"],
                    help="output-loss divergence. forward KL is mode-covering "
                         "and makes an incapable student spread its mass; "
                         "reverse is mode-seeking and lets it concentrate.")
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
    ap.add_argument("--mla-alloc", default=None,
                    help="explicit per-layer d_c: a JSON dict {layer: rank} or a "
                         "path to one (a retrieval_heads.json is accepted and its "
                         "'retrieval' key used). Mutually exclusive with "
                         "--mla-budget, which would recompute the allocation.")
    ap.add_argument("--mla-groups", default=None, metavar="JSON",
                    help="grouped latents: {layer: [[kv_heads, rank], ...]} per "
                         "layer (experiments/grouped_screen.py). One latent per "
                         "group of KV heads; overrides --mla-dc / --mla-budget. "
                         "Needs --mla-covs.")
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
                         "LoRA moved it by only 1.8%% of its norm in 2.46M "
                         "tokens, so the diagonal never really forms. ~37.7M params.")
    ap.add_argument("--train-attn", action="store_true",
                    help="dense-train ALL linear_attn and self_attn parameters "
                         "(~273M). LoRA then only matters for the FFN. Memory is "
                         "fine (~1.5 GiB of 8-bit AdamW state); the risk is data "
                         "-- 273M params against a 999k-token cache is 273 "
                         "params/token.")
    ap.add_argument("--train-norms", action="store_true",
                    help="unfreeze every normalization gain. Costs 0.05M params "
                         "(0.006%% of the model), so this is nearly free.")
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
    ap.add_argument("--vera-all", type=int, default=0, metavar="RANK",
                    help="adapt EVERY adapted matrix with VeRA at RANK -- the "
                         "KDA projections including in_proj_a, attention q/o, "
                         "and the FFN. Only the MLA latent stays dense (already "
                         "low rank), plus the parts no adapter applies to: the "
                         "diagonal, the depthwise conv, in_proj_b, norms. "
                         "~4.5 M trainable, 63x smaller than dense KDA. Rank is "
                         "nearly free here (4.49 M at 256, 4.61 M at 1024).")
    ap.add_argument("--vera-lr", type=float, default=None,
                    help="learning rate for the VeRA vectors. They start at zero and are pure scalings, so they do not move at a dense rate.")
    ap.add_argument("--lora-tiled", type=int, default=0, metavar="RANK",
                    help="adapt in_proj_a (the tiled decay projection) with "
                         "LoRA at RANK instead of training it densely. It is "
                         "37.75 M dense -- 64%% of the whole lean budget -- and "
                         "its measured update needs rank 22-26, so r64 leaves 3x "
                         "headroom over what it actually uses. 0 = dense.")
    ap.add_argument("--vera-ffn", action="store_true",
                    help="adapt mlp.* with VeRA instead of LoRA. Rank is nearly "
                         "free in VeRA (rank + out_features per layer, on a "
                         "shared frozen random pair), so this is run at a much "
                         "higher rank than LoRA would be.")
    ap.add_argument("--vera-rank", type=int, default=256)
    ap.add_argument("--vera-d-init", type=float, default=0.1)
    ap.add_argument("--freeze-ffn", action="store_true",
                    help="do not adapt mlp.* at all. The surgery damages the "
                         "attention path, and several TransMLA-style pipelines "
                         "leave the FFN alone. Measured here the FFN carries the "
                         "LARGEST relative update (0.0134 vs 0.0053 for KDA), but "
                         "magnitude is not necessity -- this flag is how to find "
                         "out whether it is doing work or merely able to move.")
    ap.add_argument("--train-data", default=None,
                    help="override the training corpus. data/fineweb_edu_long.txt\n"
                         "holds 30.0 M tokens in 3,522 documents averaging 8,519\n"
                         "tokens, against the default corpus's 550 median -- use\n"
                         "it with --doc-aware for genuine long-range structure.")
    ap.add_argument("--pct-start", type=float, default=0.05,
                    help="fraction of steps spent warming up. At 150 steps the\n"
                         "default is 7.5 steps of warmup; LR is still 79%% of\n"
                         "peak at step 50 and 26%% at step 100.")
    ap.add_argument("--rslora", action="store_true",
                    help="rank-stabilized LoRA scaling: alpha/sqrt(rank) rather\n"
                         "than alpha/rank. Decouples the update magnitude from\n"
                         "rank. alpha=4 reproduces the old scale at rank 16.")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="LoRA alpha. Default None means alpha=rank, giving\n"
                         "scale 1.0; common practice is scale 2 (alpha=2*rank,\n"
                         "or alpha=8 under rsLoRA at rank 16).")
    ap.add_argument("--doc-aware", action="store_true",
                    help="tokenize per document and draw each window INSIDE a "
                         "single document. Without it the corpus is one stream: "
                         "median document is 550 tokens, so an 8192 window "
                         "spans ~15 unrelated documents and holds no dependency "
                         "longer than ~2k. Needs a long-document corpus; build "
                         "one with src/get_long_data.py.")
    ap.add_argument("--keep-step-ckpts", action="store_true",
                    help="also write a checkpoint at every eval, not just the "
                         "best. Off by default: a merged-base run writes 1.1 "
                         "GiB per file and the root filesystem is the only one.")
    ap.add_argument("--fuse-gate-lora", action="store_true",
                    help="fold KDA's in-class rank-32 gate adapter into "
                         "in_proj_a and disable it, leaving the decay path a "
                         "single dense matrix. Without this the tiled parameter "
                         "cannot escape its rank-16 init: the parallel low-rank "
                         "path carries ~7x more of the update and is rank 1.")
    ap.add_argument("--relora-warmup", type=int, default=0,
                    help="steps of LR re-warmup after each ReLoRA merge. The "
                         "method needs a jagged schedule: a restarted adapter "
                         "enters at zero, and without re-warmup it meets "
                         "whatever the anneal has decayed to, so later merges "
                         "do progressively less. 0 reproduces the first run. "
                         "Try relora_every//4.")
    ap.add_argument("--pull-to-init", type=float, default=0.0,
                    help="decoupled decay toward the PRETRAINED weight rather "
                         "than toward zero: p -= lr*s*(p - p_init), applied to "
                         "inherited dense parameters only. MambaPEFT proposes "
                         "|W - W_pretrain|^2 in place of |W|^2 at ~1e-3. Four "
                         "arms here peak at step 100-125 and regress by 150, "
                         "which is the drift this is meant to bound. 0 = off.")
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
    ap.add_argument("--teacher", default=TEACHER,
                    help="teacher checkpoint. Must share the student's vocabulary. "
                         "Default is MERCURIUS_TEACHER (qwen3.5-27b); pass the "
                         "student's own original for self-distillation.")
    ap.add_argument("--teacher-bits", type=int, default=4, choices=[4, 16],
                    help="load the teacher NF4 (4) or bf16 (16). Its lm_head "
                         "and embeddings stay bf16 either way.")
    ap.add_argument("--student-bits", type=int, default=4, choices=[4, 16],
                    help="after surgery, quantize every FROZEN Linear of the "
                         "student to NF4. Trainable weights (adapters, MLA "
                         "latents, per-head maps, norms, gates) are never "
                         "quantized, and neither are the tied embeddings or the "
                         "decay/erase/write gate projections (numerically "
                         "sensitive: they feed a cumulative log-decay).")
    ap.add_argument("--taid-space", default="auto", choices=["auto", "logit", "prob"],
                    help="TAID interpolation space. auto picks the one that is "
                         "not degenerate for --divergence (forward: logit, the "
                         "paper's; reverse: prob, i.e. skew reverse KL). The "
                         "other choice reduces TAID to a loss-scale ramp and is "
                         "refused unless named explicitly.")
    ap.add_argument("--taid-start", type=float, default=0.4)
    ap.add_argument("--taid-end", type=float, default=1.0)
    ap.add_argument("--taid-alpha", type=float, default=5e-4)
    ap.add_argument("--taid-beta", type=float, default=0.99)
    ap.add_argument("--taid-linear", action="store_true",
                    help="linear t_start -> t_end, no adaptive term")
    ap.add_argument("--scalenorm", action="store_true",
                    help="stage A': replace every FOLDED RMSNorm with ScaleNorm, "
                         "one learned scalar each (64 parameters instead of "
                         "163,840). Bitwise exact at init -- stage A already left "
                         "those gains at 1. Unfoldable norms (model.norm, "
                         "q/k_norm, linear_attn.norm) stay per-channel.")
    ap.add_argument("--mtp", type=int, default=0, metavar="K",
                    help="conv multi-token-prediction head (models/mtp_conv.py) "
                         "predicting t+2..t+K+1 from the final hidden states; 0 = "
                         "off. The trunk's own lm_head keeps t+1 untouched. Head j "
                         "is distilled against the teacher's logits at t+j, which "
                         "the main loss already computes.")
    ap.add_argument("--mtp-weight", type=float, default=0.1,
                    help="weight on the MTP loss (mean over heads)")
    ap.add_argument("--mtp-lr", type=float, default=1e-3,
                    help="own LR group for the head. At init z_j = h exactly, so "
                         "until its zero-init output projection opens, the MTP "
                         "loss asks the TRUNK to predict t+2.. with the t+1 head; "
                         "at the dense 3e-5 that would last most of a run")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--mem-cap-gb", type=float, default=56.0,
                    help="hard cap on this process's CUDA allocator. Unified "
                         "memory is shared with vLLM (~42 GB) and others; "
                         "exhausting it freezes the machine instead of raising")
    ap.add_argument("--min-avail-gb", type=float, default=12.0,
                    help="stop cleanly if system MemAvailable falls below this")
    ap.add_argument("--gpu-temp-pause", type=float, default=84.0)
    ap.add_argument("--gpu-temp-resume", type=float, default=80.0)
    ap.add_argument("--acpi-temp-pause", type=float, default=90.0)
    ap.add_argument("--acpi-temp-resume", type=float, default=85.0)
    ap.add_argument("--eval-gpu-start", type=float, default=75.0,
                    help="before each eval forward (one uninterrupted burst, "
                         "~+15C for the 27B at 8192), wait until the GPU is at "
                         "or below this")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to sleep after every step (duty cycle)")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    # Argument consistency, checked BEFORE anything touches the GPU.
    # (reverse/js with --ce-beta used to be refused here: the data term was the
    # chosen divergence against a near-one-hot, which in reverse mode collapses
    # the student to a point mass. The data term is now excess CE in every mode,
    # which has no such collapse -- see _chunk_div_terms.)
    if a.taid:
        if a.divergence == "js":
            raise SystemExit("--taid with --divergence js is not derived; use "
                             "forward or reverse")
        if a.divergence == "jeffreys" and a.taid_space == "auto":
            raise SystemExit(
                "--taid with --divergence jeffreys has no clean space: in prob "
                "space the FORWARD term becomes a loss-scale ramp, in logit space "
                "the REVERSE term does (taid_space_for). Name one explicitly with "
                "--taid-space logit|prob, knowing which half TAID then acts on.")
        want = taid_space_for(a.divergence)
        if a.taid_space == "auto":
            a.taid_space = want
        elif a.taid_space != want:
            print(f"  WARNING: --taid-space {a.taid_space} with --divergence "
                  f"{a.divergence} makes TAID an exact loss-scale ramp, not an "
                  f"interpolated target (taid_space_for). Running it because "
                  f"you named it.", flush=True)
    if a.relora_every and not a.save_merged_bases:
        raise SystemExit(
            "--relora-every merges adapters into the frozen bases mid-run, and "
            "that merge is NOT replayable: it depends on the trained factors at "
            "the moment of merging, which are then reset. Saving adapters only "
            "would produce a checkpoint that cannot be rebuilt at all. Re-run "
            "with --save-merged-bases (about 0.95 GB per checkpoint).")

    from mercurius.paths import ensure_dirs
    from mercurius import guard
    ensure_dirs()
    _frac, _tot = guard.cap_cuda_memory(a.mem_cap_gb)
    print(f"  memory: CUDA allocator capped at {a.mem_cap_gb:.0f} of {_tot:.0f} GiB "
          f"unified; system MemAvailable {guard.mem_available_gb():.1f} GiB "
          f"(floor {a.min_avail_gb:.0f})", flush=True)
    if guard.mem_available_gb() < a.mem_cap_gb + a.min_avail_gb:
        raise SystemExit(
            f"refusing to start: MemAvailable {guard.mem_available_gb():.1f} GiB "
            f"is below cap {a.mem_cap_gb:.0f} + floor {a.min_avail_gb:.0f}. Other "
            f"processes share this unified memory; free some or lower --mem-cap-gb.")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    # DIFFERENT corpora: training on the eval set would make perplexity measure
    # memorization instead of recovery.
    train_path = a.train_data or TRAIN_DATA
    if a.doc_aware:
        if a.synth_data and a.state_passing:
            # interleave, do not append: see interleave_corpora. Tokenise the
            # natural corpus once, here, instead of letting the generic path
            # below do it as well -- doing both cost ~6 min of startup.
            _a = _docs_and_spans(tok, train_path, a.seq)
            _b = _docs_and_spans(tok, a.synth_data, a.seq)
            _n_syn = sum(int(x.numel()) for x in _b)
            train_ids, doc_spans = interleave_corpora(_a, _b, a.seq)
            print(f"  synthetic multi-item data: {len(_b)} documents interleaved "
                  f"through {len(_a)}, {_n_syn/int(train_ids.numel()):.1%} of TOKENS "
                  f"(sequential sampling reaches them only if interleaved)",
                  flush=True)
            print(f"  document-aware: {len(_a)} natural docs, "
                  f"{sum(1 for x in _a if x.numel() >= a.seq)} at least {a.seq} tokens",
                  flush=True)
        else:
            train_ids, doc_spans = tokenize_by_document(tok, train_path, a.seq)
        if a.synth_data and not a.state_passing:
            # Appended, not substituted. batches() picks a span UNIFORMLY, so the
            # synthetic share of draws is n_synth_spans / total_spans -- not the
            # token share, which would be far smaller since these documents are
            # only ~9k tokens each.
            s_ids, s_spans = tokenize_by_document(tok, a.synth_data, a.seq)
            off = int(train_ids.numel())
            n_nat = len(doc_spans or [])
            doc_spans = (doc_spans or []) + [(s + off, e + off) for s, e in s_spans]
            train_ids = torch.cat([train_ids, s_ids])
            # Report the share that will ACTUALLY apply. With --state-passing the
            # sampler is sequential and doc spans are never consulted, so the
            # governing quantity is the TOKEN share, not the span share -- and the
            # two differ by 2x here. Printing the span share under state passing
            # would overstate the mix.
            _tok_share = int(s_ids.numel()) / int(train_ids.numel())
            _span_share = len(s_spans) / max(len(doc_spans), 1)
            if a.state_passing:
                print(f"  synthetic multi-item data: {len(s_spans)} documents, "
                      f"{_tok_share:.1%} of TOKENS (state passing samples "
                      f"sequentially, so doc spans do not govern the mix)",
                      flush=True)
            else:
                print(f"  synthetic multi-item data: {len(s_spans)} spans added to "
                      f"{n_nat}; {_span_share:.1%} of sampled "
                      f"windows will require multi-fact retention", flush=True)
    else:
        train_ids = tok(open(train_path).read(), return_tensors="pt").input_ids[0]
        doc_spans = None
    eval_ids = tok(open(EVAL_DATA).read(), return_tensors="pt").input_ids[0]
    print(f"train {len(train_ids):,} tok ({train_path.split('/')[-1]}) | "
          f"eval {len(eval_ids):,} tok (wikitext, held out)", flush=True)
    print(f"dial={a.dial} seeded={a.seed_decay} steps={a.steps} seq={a.seq}",
          flush=True)
    print(f"epochs over train corpus: "
          f"{a.steps * a.seq / len(train_ids):.3f}", flush=True)

    def load_teacher():
        """The teacher may be a different, larger member of the family, with its
        own output head (see _chunk_div_terms). Loaded AFTER the student's
        surgery and quantization, so the student's transient bf16 peak and the
        teacher never coexist -- on a shared unified-memory board the order of
        these two allocations is the difference between fitting and not."""
        # The teacher may be a different, larger member of the family. It gets its
        # own output head (see _chunk_div_terms) and is loaded NF4 by default: a
        # 27B in bf16 is 54 GB before activations. bitsandbytes leaves lm_head and
        # the embeddings unquantized, so the logits it supervises with are bf16.
        if a.teacher_bits == 4:
            # Streamed straight into NF4, one tensor at a time (stream_nf4). The
            # from_pretrained + BitsAndBytesConfig path staged >39 GiB loading the
            # 27B and was OOM-killed on this shared unified-memory board; streaming
            # peaks at 14.8 GiB. NF4 weights are bit-identical to that path.
            from mercurius.models.stream_nf4 import load_nf4
            teacher = load_nf4(a.teacher).eval()
        else:
            teacher = AutoModelForCausalLM.from_pretrained(
                a.teacher, dtype=torch.bfloat16, device_map="cuda").eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        _tv = teacher.get_output_embeddings().weight.shape[0]
        if _tv != len(tok) and _tv < len(tok):
            raise SystemExit(f"teacher vocabulary {_tv} is smaller than the "
                             f"tokenizer's {len(tok)}; teacher and student must share it")
        print(f"  teacher: {a.teacher.rstrip('/').split('/')[-1]} "
              f"({'NF4' if a.teacher_bits == 4 else 'bf16'}), "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB resident", flush=True)
        return teacher

    if a.build_cache:
        teacher = load_teacher()
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
    if a.gdn2:
        # BEFORE adapters. from_kda copies submodules by state_dict, and a
        # LoRA/VeRA-wrapped projection has a different structure, so lifting an
        # already-adapted model would throw or silently drop the adapter.
        from mercurius.models.gdn2 import convert_to_gdn2
        convert_to_gdn2(student)
    if a.scalenorm:
        # after the GDN-2 lift, before adapters and the MLA conversion: it only
        # swaps norm modules, so nothing downstream sees a different structure
        from mercurius.surgery.scalenorm import convert_to_scalenorm
        convert_to_scalenorm(student)
    if a.decay_phase is not None:
        from mercurius.models.kda import Qwen3_5KDAGatedDeltaNet as _KDA
        _n = 0
        for _l in (student.model.language_model if hasattr(student.model, "language_model")
                   else student.model).layers:
            _la = getattr(_l, "linear_attn", None)
            if isinstance(_la, _KDA):
                _la.enable_decay_phase(a.decay_phase); _n += 1
        print(f"  decay-tied phase enabled on {_n} linear-attention layers "
              f"(c init {a.decay_phase:g}); MLA layers stay NoPE", flush=True)

    if a.state_passing:
        n_sp = enable_state_passing(student)
        print(f"  state passing enabled on {n_sp} linear-attention (GDN-2) layers", flush=True)

    if a.grad_checkpoint:
        # Non-reentrant: with the embeddings frozen, the reentrant variant sees
        # no input requiring grad and silently returns no gradients.
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        student.config.use_cache = False
        print("  gradient checkpointing enabled", flush=True)

    # prior adapters load BEFORE conversion so their delta is merged into the
    # factorization rather than discarded by it
    # Built ONCE and used by BOTH injection sites. When --init-adapters is set,
    # LoRA is injected early to shape the model for the load; with patterns
    # matched by endswith, the later call then finds nothing left to wrap and is
    # a no-op. Any rule change applied only to the later call is therefore
    # silently discarded -- which is how --kda-rank printed "rank -> 64" and
    # trained rank 16.
    rules = LORA_RULES
    if a.freeze_ffn or a.vera_ffn:
        rules = [(r[0], 0) + tuple(r[2:]) if r[0].startswith('mlp.') else r
                 for r in rules]
        print('  FFN frozen: no adapter on mlp.*', flush=True)
    VERA_ALL = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                "linear_attn.out_proj", "linear_attn.in_proj_a",
                # GDN-2's channel-wise erase/write gates. Tiled at init (every
                # channel in a head shares one row), which is the structure the
                # architecture exists to escape, so they must be adapted.
                "linear_attn.in_proj_be", "linear_attn.in_proj_bw",
                "self_attn.q_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    if a.vera_all:
        # every LoRA rule becomes rank 0; VeRA is installed on the same targets
        # after the init adapters are folded in
        rules = [(r[0], 0) + tuple(r[2:]) for r in rules]
        print(f"  ALL-VeRA at rank {a.vera_all}: only the MLA latent stays dense",
              flush=True)
    if a.lora_tiled:
        rules = list(rules) + [("linear_attn.in_proj_a", a.lora_tiled, 4.0)]
        print(f"  in_proj_a adapted with LoRA r{a.lora_tiled} "
              f"(37.75 M dense -> {a.lora_tiled*(1024+2048)*18/1e6:.2f} M)", flush=True)
    if a.kda_rank:
        # chain from `rules`, NOT from LORA_RULES: starting over from the
        # constant discards any earlier modification. --freeze-ffn worked alone
        # and --kda-rank worked alone, and together the freeze was dropped, so
        # the FFN trained 10.62 M of adapters in a run whose whole point was
        # that it had none.
        rules = [((rule[0], a.kda_rank) + tuple(rule[2:]))
                 if rule[0].startswith("linear_attn.") else rule
                 for rule in rules]
        print(f"  KDA projection LoRA rank -> {a.kda_rank}", flush=True)

    if a.init_adapters:
        # Inject at the CHECKPOINT's ranks, not the recipe's. They diverge the
        # moment any rule changes, and load_state_dict forgives missing keys but
        # not mismatched shapes, so the run dies at startup rather than loading
        # a different model -- which is the good failure, but only if the ranks
        # are read from the file.
        _sd = torch.load(a.init_adapters, map_location="cpu")
        ckpt_rules = rules_from_checkpoint(_sd)
        inject_lora(student, ckpt_rules, verbose=False,
                    alpha=a.lora_alpha, rslora=a.rslora)
        freeze_base(student)
        student.load_state_dict({k: v.cuda() for k, v in _sd.items()}, strict=False)
        print(f"  init from {a.init_adapters.split('/')[-1]} "
              f"({len(_sd)} tensors)", flush=True)
        want = {r[0]: r[1] for r in rules}
        have = dict(ckpt_rules)
        if any(want.get(k) not in (None, v) for k, v in have.items()) or \
           any(k not in have for k in want):
            # fold the loaded delta into the bases FIRST, then resize; the other
            # order discards the prior init silently
            diffs = {k: (have.get(k), want.get(k)) for k in set(have) | set(want)
                     if have.get(k) != want.get(k)}
            print(f"  init checkpoint ranks differ from the recipe: {diffs}",
                  flush=True)
            merge_and_restart(student)
            resize_lora(student, rules)
            # rank 0 means REMOVE the adapter. resize_lora cannot express that
            # -- its guard is `if r and r != mod.rank`, and 0 is falsy, so a
            # rank-0 rule silently leaves the adapter at whatever rank the
            # checkpoint had. --freeze-ffn looked like it worked (the flag
            # printed, the rank diff printed) and trained 5.31 M of FFN adapters
            # anyway.
            drop = [r[0] for r in rules if r[1] == 0]
            if drop:
                n_dropped = unwrap_lora(student, drop)
                if n_dropped:
                    print(f"  removed {n_dropped} adapters at rank 0 "
                          f"({', '.join(drop)})", flush=True)

    if a.fuse_gate_lora:
        fuse_gate_lora(student)

    if a.vera_all:
        merge_and_restart(student)                 # fold the init delta in first
        n_un = unwrap_lora(student, VERA_ALL)
        inject_vera(student, [(p, 1) for p in VERA_ALL],
                    rank=a.vera_all, d_init=a.vera_d_init)
        print(f"  unwrapped {n_un} LoRA modules before installing VeRA", flush=True)

    if a.vera_ffn:
        # Fold whatever the init checkpoint put in the FFN adapters into the
        # bases, drop the LoRA wrappers, then install VeRA on the bare Linears.
        # Skipping the merge would silently discard the initialization.
        merge_and_restart(student)
        n_un = unwrap_lora(student, ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
        inject_vera(student, [("mlp.gate_proj", 1), ("mlp.up_proj", 1),
                              ("mlp.down_proj", 1)],
                    rank=a.vera_rank, d_init=a.vera_d_init)
        print(f"  swapped {n_un} FFN adapters from LoRA to VeRA", flush=True)

    mla_latents = []
    if (a.mla_energy is not None or a.mla_dc is not None or a.mla_budget is not None
            or a.mla_groups):
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
        _alloc = None
        if a.mla_alloc:
            import json as _j, os as _o
            _alloc = _j.load(open(a.mla_alloc)) if _o.exists(a.mla_alloc) \
                else _j.loads(a.mla_alloc)
            _alloc = _alloc.get("retrieval", _alloc) if isinstance(_alloc, dict) else _alloc
        _groups = None
        if a.mla_groups:
            _gj = json.load(open(a.mla_groups))["groups"]
            _groups = {int(l): [(list(h), int(r)) for h, r in g] for l, g in _gj.items()}
            print(f"  grouped latents from {a.mla_groups.split('/')[-1]}: "
                  f"{ {l: [r for _, r in g] for l, g in _groups.items()} } "
                  f"(total {sum(r for g in _groups.values() for _, r in g)})", flush=True)
        info = convert_to_mla(student, alloc=_alloc, d_c=a.mla_dc, budget=a.mla_budget,
                              energy=a.mla_energy if a.mla_energy else 0.95,
                              covs=covs, groups=_groups)
        for l in get_trunk(student).layers:
            sa = getattr(l, "self_attn", None)
            if sa is not None and hasattr(sa.k_proj, "latent"):
                mla_latents.append(sa.k_proj.latent)

    inject_lora(student, rules, alpha=a.lora_alpha, rslora=a.rslora)
    # Substring matching against parameter names. LoRA-wrapped modules expose
    # their frozen base as "<mod>.base.weight", so a module-prefix pattern like
    # "linear_attn." unfreezes the dense base AND its adapter together -- which
    # is why no adapter merging is needed to switch a module from LoRA to dense.
    # after inject_lora, so lam scales (base + LoRA delta), not the frozen base
    if a.layerscale:
        install_layerscale(student)
    also = ("lora_A", "lora_B", "a_lora_A", "a_lora_B", "A_log", "dt_bias",
            "vera_d", "vera_b")
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
        keep = ("in_proj_b.weight", "conv1d.")
        # in_proj_a is either dense OR adapted, never both: leaving it in the
        # dense set alongside its own adapter is the parallel-path problem that
        # made the KDA gate unmeasurable (a dense matrix and a low-rank matrix
        # competing on the same output, the low-rank one winning 7 to 1).
        also = also + keep + (() if (a.lora_tiled or a.vera_all)
                              else ("in_proj_a.weight",))
    if a.train_attn:
        also = also + ("linear_attn.", "self_attn.")
    if a.train_norms:
        also = also + ("norm",)
    if a.decay_phase is not None:
        also = also + ("pe_c",)
    n_tr = freeze_base(student, also_train=also)

    if a.fuse_gate_lora:
        # freeze_base matches substrings and "lora_A" matches "a_lora_A", so the
        # fused factors cannot be excluded through `also`. They are inert either
        # way (lora_rank=0 means _decay skips them, so no gradient reaches them)
        # but leaving them trainable puts 1.77 M dead parameters in the optimizer
        # and in the reported count.
        dead = 0
        for nm, prm in student.named_parameters():
            if "a_lora_" in nm:
                prm.requires_grad_(False)
                dead += prm.numel()
        n_tr -= dead
        print(f"  froze {dead/1e6:.2f} M fused gate-adapter params", flush=True)
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

    if a.per_head_q:
        # After convert_to_mla replaced k_proj/v_proj, and BEFORE the parameter
        # groups are collected. Installed after the collection, R never reaches
        # the optimizer: it stays at identity for the whole run while the log
        # cheerfully reports it as trainable.
        from mercurius.surgery.perhead_q import install_per_head_q
        install_per_head_q(student)
        # n_tr was totalled before this point, so add the new parameters or the
        # log under-reports by 3.15 M. The optimizer is correct either way --
        # trainable_parameters walks the model below -- but a trainable count
        # that silently disagrees with the model is how every earlier
        # accounting bug here went unnoticed.
        n_tr += sum(p.numel() for n, p in student.named_parameters()
                    if n.endswith(".R") and p.requires_grad)

    if a.mtp:
        from mercurius.models.mtp_conv import ConvMTPHead
        _cfg = getattr(student.config, "text_config", student.config)
        student.mtp_head = ConvMTPHead(d_model=_cfg.hidden_size, k=a.mtp)
        for prm in student.mtp_head.parameters():
            prm.requires_grad_(True)
        _nm = student.mtp_head.n_params()
        n_tr += _nm
        print(f"  conv MTP head: K={a.mtp} (t+2..t+{a.mtp + 1}), {_nm:,} params, "
              f"receptive field {student.mtp_head.receptive_field}, identity at "
              f"init; weight {a.mtp_weight:g}, lr {a.mtp_lr:g}", flush=True)

    # Guard, placed AFTER every mechanism is installed.
    #
    # Placed right after freeze_base it was a FALSE POSITIVE: per-head query
    # maps are installed ~90 lines later, so the check ran before R existed
    # and killed a valid run. A guard that fires on correct code is worse
    # than no guard. Adding "pe_c" to freeze_base's DEFAULT did nothing
    # because this call passes `also` explicitly, so the default is dead code --
    # the same shape of mistake as fixing the parameter grouping inside an elif
    # branch that never runs. phase33 and phase2 both trained a FIXED c for 150
    # steps and were read as tests of a learned phase.
    #
    # A requested mechanism whose parameter is not trainable is a silent null
    # result, so assert the observable instead of trusting the code to read
    # right. Cheap, and it fails at startup rather than at analysis time.
    for flag, needle, what in (
            (a.decay_phase is not None, "pe_c", "--decay-phase"),
            (a.per_head_q, ".R", "--per-head-q"),
            (a.gdn2, "in_proj_be", "--gdn2"),
            (bool(a.mtp), "mtp_head.", "--mtp"),
            (a.scalenorm and a.train_norms, "layernorm.weight", "--scalenorm")):
        if not flag:
            continue
        live = [n for n, p in student.named_parameters()
                if needle in n and p.requires_grad]
        if not live:
            raise SystemExit(
                f"{what} was requested but no parameter matching '{needle}' is "
                f"trainable, so the mechanism cannot move and the run would "
                f"report a null result for an untested change. Check that the "
                f"name is in `also` above, and that nothing froze it after.")
        print(f"  {what}: {len(live)} trainable tensors matching '{needle}'",
              flush=True)

    if a.student_bits == 4:
        # LAST structural change, after everything that reads weights: the MLA
        # factorization, adapter merges and the GDN-2 tiling all need the bases
        # in high precision, which is why every stage runs before this.
        from mercurius.models.quantize import quantize_frozen_nf4
        nq, nkeep = quantize_frozen_nf4(student)
        torch.cuda.empty_cache()
        print(f"  student: {nq} frozen Linear -> NF4, {nkeep} kept bf16; "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB resident "
              f"(teacher + student)", flush=True)

    teacher = load_teacher()
    pacer = guard.ThermalPacer(a.gpu_temp_pause, a.gpu_temp_resume,
                               a.acpi_temp_pause, a.acpi_temp_resume)
    print(f"  thermal pacer on {pacer.attach(student) + pacer.attach(teacher)} "
          f"decoder layers (pause {a.gpu_temp_pause:g}C -> resume "
          f"{a.gpu_temp_resume:g}C, checked before every layer)", flush=True)
    print(f"  after teacher: {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, "
          f"system MemAvailable {guard.mem_available_gb():.1f} GiB", flush=True)

    params = trainable_parameters(student)
    print(f"  trainable: {n_tr/1e6:.2f} M params in {len(params)} tensors", flush=True)

    # One group by default, which means LoRA adapters train at the DENSE-safe
    # rate. That is 5-10x below normal LoRA practice, and it silently handicaps
    # every LoRA-heavy arm: a surface ablation run this way measures the
    # learning rate, not the surface.
    if a.vera_lr:
        _is_v = lambda n: "vera_d" in n or "vera_b" in n
        _is_r = lambda n: n.endswith(".R")
        vp = [p for n, p in student.named_parameters() if p.requires_grad and _is_v(n)]
        # The per-head query map gets its OWN rate, for a reason that is neither
        # "dense" nor "adapter":
        #   at the dense rate 3e-5, AdamW's displacement ceiling over 150 steps
        #   is 0.0045 per element. Measured on gdn2phq, R ended 0.42% off
        #   identity with a max off-diagonal of 0.0017 -- the mechanism was
        #   never exercised, so its +0.4 EM tested nothing.
        #   at the VeRA rate 1e-2 the ceiling is 1.5 per element, LARGER than
        #   the identity diagonal R is initialised to, so R can be erased.
        # 1e-3 gives a ceiling of 0.15: enough for off-diagonals of order
        # 0.01-0.1, which separates the heads without destroying the query.
        rp = [p for n, p in student.named_parameters() if p.requires_grad and _is_r(n)]
        _is_m = lambda n: n.startswith("mtp_head.")
        mp = [p for n, p in student.named_parameters() if p.requires_grad and _is_m(n)]
        dp = [p for n, p in student.named_parameters()
              if p.requires_grad and not _is_v(n) and not _is_r(n) and not _is_m(n)]
        groups = [{"params": dp, "lr": a.lr}, {"params": vp, "lr": a.vera_lr}]
        max_lr = [a.lr, a.vera_lr]
        if rp:
            groups.append({"params": rp, "lr": a.phq_lr})
            max_lr.append(a.phq_lr)
        if mp:
            groups.append({"params": mp, "lr": a.mtp_lr})
            max_lr.append(a.mtp_lr)
            print(f"  MTP head group: {sum(p.numel() for p in mp):,} params "
                  f"@ {a.mtp_lr:g}", flush=True)
        print(f"  {2 + (1 if rp else 0)} groups: {sum(p.numel() for p in dp)/1e6:.2f} M dense "
              f"@ {a.lr:g}, {sum(p.numel() for p in vp)/1e6:.3f} M VeRA "
              f"@ {a.vera_lr:g}"
              + (f", {sum(p.numel() for p in rp)/1e6:.2f} M per-head-q "
                 f"@ {a.phq_lr:g}" if rp else ""), flush=True)
    elif a.lora_lr:
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
    # cycle_momentum defaults to True and, for Adam-family optimizers, cycles
    # BETA1 through the `betas` entry -- it overwrote our (0.9, 0.95) with
    # (0.95, 0.95) at construction and then swept beta1 between 0.85 and 0.95.
    # Every run before 2026-09-13 trained with a cycled beta1 nobody asked for.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=a.steps, pct_start=a.pct_start,
        cycle_momentum=False)

    # Pull-to-init needs the pretrained value of every inherited dense weight.
    # Adapters are EXCLUDED: their "init" is kaiming/zeros, not a pretrained
    # operator, so pulling them back is ordinary weight decay and not what this
    # is for. Snapshot in bf16 alongside the weights -- ~0.56 GB at 280 M dense.
    # sentinel, not 0: at 0 the warmup window covers the first steps of
    # training, stacking on OneCycle's own warmup before any merge exists
    last_merge = -10**9
    pull_ref = []
    if a.pull_to_init > 0:
        skip = ("lora_A", "lora_B", "a_lora_A", "a_lora_B", "ls_lambda")
        for nm, prm in student.named_parameters():
            if prm.requires_grad and not any(k in nm for k in skip):
                pull_ref.append((prm, prm.detach().clone()))
        mb = sum(r.numel() * r.element_size() for _, r in pull_ref) / 1e9
        print(f"  pull-to-init {a.pull_to_init:g} on {len(pull_ref)} inherited "
              f"tensors ({mb:.2f} GB reference copy)", flush=True)

    taid = (TAIDSchedule(a.steps, a.taid_start, a.taid_end, a.taid_alpha,
                         a.taid_beta, adaptive=not a.taid_linear)
            if a.taid else None)
    if taid is not None:
        print(f"  TAID: {a.taid_space} space, t {a.taid_start:g} -> "
              f"{a.taid_end:g}, {'linear' if a.taid_linear else 'adaptive'}",
              flush=True)
    hist = {"loss": [], "div": [], "data": [], "taid_t": [], "eval": []}
    _best = {"ppl": float("inf"), "step": -1}

    def evaluate(step):
        student.eval()
        was_sp = a.state_passing
        if was_sp:
            enable_state_passing(student, False)   # eval without carried state
        row = {"step": step}
        print(f"  [eval @ {step:>4}]", flush=True)
        for n in (2048, 8192):
            m = ce_and_topk(student, eval_ids, n, teacher=teacher,
                            before_forward=lambda: guard.cool_to(
                                a.eval_gpu_start, a.acpi_temp_resume,
                                log=lambda msg: print(msg, flush=True)))
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
        # Keep the BEST checkpoint, not every one. Measured across seven arms,
        # every run peaks early and then degrades by 0.8-2.9%: gate8kv2 18.219
        # at step 50 against 18.630 at 150, relora 18.144 against 18.676. Eval
        # is bit-deterministic (spread 0.0000% over five repeats and three
        # reloads), so that degradation is real and selecting on it is sound.
        #
        # This also SHRINKS the footprint: one overwritten best plus one final,
        # instead of one file per eval. A merged-base run writes 1.1 GiB per
        # checkpoint, so four evals cost 4.4 GiB where this costs 1.1 GiB.
        cur = row[8192]["ppl"]
        improved = cur < _best["ppl"]
        if improved:
            _best.update(ppl=cur, step=step)
        try:
            st = os.statvfs("/")
            if st.f_bavail * st.f_frsize > 8 * 2**30:      # keep 8 GiB headroom
                ck = (str(CKPT_DIR / f"adapters-{a.tag}-best.pt"))
                if improved:
                    save_trainable(student, ck, a.save_merged_bases)
                    print(f"    best so far: ppl@8192 {cur:.3f} -> {ck.split('/')[-1]}",
                          flush=True)
                elif a.keep_step_ckpts:
                    save_trainable(student, ck.replace("-best.pt", f"-step{step}.pt"),
                                   a.save_merged_bases)
                # Overwritten each eval, so it stays ~1.1 GiB rather than growing.
                if step > 0 and _resume_ctx:
                    save_resume(str(CKPT_DIR / f"resume-{a.tag}.pt"),
                                student, _resume_ctx["opt"], _resume_ctx["sched"],
                                step, _resume_ctx["gens"], a.save_merged_bases,
                                _resume_ctx["taid"])
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
    _op_announced = False
    # On-policy rollouts must BE retrieval episodes. Anchored at an arbitrary
    # offset they were free continuations of generic prose -- no facts to retain,
    # no restatement to produce -- and the arm came last on retrieval (62.3% vs
    # 76.9% without it, findings 0.5b). Anchoring at the LAST restatement header
    # inside the window makes the student generate the codes it must have
    # retained, with the whole window as context, and corrects it there.
    from mercurius.recovery.synth_recall import SUMMARY_HEAD
    _op_mark = tok(" " + SUMMARY_HEAD, add_special_tokens=False).input_ids
    _op_mark_t = torch.tensor(_op_mark)
    _op_skipped = 0
    _op_inelig = 0
    _op_steps = 0
    _bg_op = random.Random(20260920)   # on-policy coin, independent of
                                       # the window sampler so a matched
                                       # control sees identical windows
    _resume_ctx = {"opt": opt, "sched": sched, "gens": {"mix": _g, "batch": _bg},
                   "taid": taid}
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
        if doc_spans is not None:
            n = cache["n_tokens"]
            before = len(doc_spans)
            doc_spans = [(s, e) for s, e in doc_spans if e <= n]
            print(f"  doc spans trimmed to the cached prefix: "
                  f"{before} -> {len(doc_spans)}", flush=True)
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
        if taid is not None and _rs.get("taid"):
            taid.load_state_dict(_rs["taid"])
        start_step = int(_rs["step"])
        print(f"  RESUMED from {a.resume.split('/')[-1]} at step {start_step}/{a.steps} "
              f"(optimizer moments, LR position and sampler RNG restored)", flush=True)
        del _rs
    gen = (sequential_batches(sample_ids, sl, a.steps - start_step) if a.state_passing
           else batches(sample_ids, sl, a.steps - start_step, align=align,
                       gen=_bg, spans=doc_spans))
    for step, (batch, new_doc, batch_offset) in enumerate(gen, start=start_step + 1):
        # TAID's target walks from the student's own distribution to the
        # teacher's as training proceeds; ignored by the other modes
        _taid_lam = taid.t if taid is not None else 1.0
        guard.wait_until_cool(a.gpu_temp_pause, a.gpu_temp_resume,
                              a.acpi_temp_pause, a.acpi_temp_resume,
                              log=lambda m: print(m, flush=True))
        _av = guard.mem_available_gb()
        if _av < a.min_avail_gb:
            print(f"  STOPPING at step {step}: MemAvailable {_av:.1f} GiB below "
                  f"the {a.min_avail_gb:.0f} GiB floor. Resume from the last "
                  f"resume file.", flush=True)
            break
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
                (student.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                 if want else student.gradient_checkpointing_disable())
                student.config.use_cache = False
                _ckpt_on[0] = want
        L = x.shape[1]
        # ---------------------------------------------------------- on-policy
        # GKD: on a fraction of steps, distil on what the STUDENT would actually
        # say rather than on corpus text. The divergence is then measured on
        # positions the student itself produced, so it learns to recover from its
        # own errors -- the failure teacher forcing cannot present. Measured on
        # this model: converted arms have BETTER teacher-forced likelihood than
        # the original and far worse exact match on multi-value retrieval, which
        # is exposure bias and nothing else.
        op_lo = 0
        if a.on_policy > 0.0:
            # On-policy rollouts must BE retrieval episodes. Anchored at an
            # arbitrary offset they were free continuations of generic prose -- no
            # facts to retain, no restatement to produce -- and that arm came last
            # on retrieval, 62.3% against 76.9% without it (findings 0.5b).
            #
            # Anchoring at the LAST restatement header in the window keeps the
            # whole preceding context AND makes the generated tokens the values
            # the student must have retained, so the correction lands on the
            # failure mode. A window with no header is natural text with nothing
            # to retrieve, and stays teacher-forced.
            _w = x[0].cpu()
            _hit = -1
            if _w.numel() > _op_mark_t.numel():
                _eq = (_w.unfold(0, _op_mark_t.numel(), 1) == _op_mark_t).all(dim=1).nonzero()
                if _eq.numel():
                    _hit = int(_eq[-1]) + _op_mark_t.numel()
            # Draw AFTER establishing eligibility. Drawing first spends the
            # budget on windows with no header: under --state-passing only ~19%
            # of windows carry a synthetic span, so an 0.5 rate became an
            # effective ~0.1 and, before the interleave fix, exactly 0.0.
            if _hit < 64 or _hit >= L - 8:
                _op_inelig += 1
            elif _bg_op.random() >= a.on_policy:
                _op_skipped += 1
            else:
                _op_steps += 1
                plen = _hit
                was_ckpt = getattr(student, "is_gradient_checkpointing", False)
                if was_ckpt:
                    student.gradient_checkpointing_disable()
                # Suspend state passing across the rollout. During incremental
                # decoding the recurrent state must come from the decode cache;
                # _carry overrides it (kda_model: `if self._state_passing and
                # self._carry is not None`), so every generated token would
                # restart from the stale carried state and the rollout would be
                # garbage -- silently, since it still returns tokens. The flag is
                # all that is toggled, so _carry survives for the training
                # forward that follows.
                if a.state_passing:
                    enable_state_passing(student, False)
                student.config.use_cache = True
                student.eval()
                with torch.no_grad():
                    x = student.generate(
                        x[:, :plen], max_new_tokens=a.on_policy_gen,
                        do_sample=True, temperature=1.0, top_p=1.0,
                        pad_token_id=tok.eos_token_id, use_cache=True)
                student.train()
                student.config.use_cache = False
                if a.state_passing:
                    enable_state_passing(student, True)
                if was_ckpt:
                    student.gradient_checkpointing_enable()
                L = x.shape[1]
                # token at index plen is the first the student chose, so the
                # position that PREDICTS it is plen-1
                op_lo = plen - 1
                if not _op_announced:
                    print(f"  on-policy: anchored at a restatement header "
                          f"(token {plen}), scored {L - 1 - op_lo} of {L - 1} "
                          f"positions, all student-generated", flush=True)
                    _op_announced = True
        s_logits = t_logits = None
        if a.live_teacher:
            # Exact full-vocabulary KL against a teacher conditioned on the
            # student's actual prefix. No cache, so no support truncation, no
            # block alignment, and windows may start anywhere in the corpus.
            with torch.no_grad():
                h_t = get_trunk(teacher)(input_ids=x).last_hidden_state[0]
            h_s = get_trunk(student)(input_ids=x).last_hidden_state[0]
            W_s = student.get_output_embeddings().weight
            W_t = teacher.get_output_embeddings().weight
            # position p predicts token p+1, so the last position has no target
            # and is dropped -- 1 of 8192, and it keeps the data term defined
            nxt = x[0, 1:]
            tot, n = 0.0, 0
            if a.mtp:
                # Main loss and all MTP heads together (_chunk_mtp_terms): the
                # teacher's log-softmax is formed once per chunk for every
                # offset, and the student's K+1 row sets go through one GEMM.
                z = student.mtp_head(h_s.unsqueeze(0))[0]      # (T, K, d)
                m_tot, m_n = 0.0, 0
                for i in range(op_lo, L - 1, a.fullkl_chunk):
                    j = min(i + a.fullkl_chunk, L - 1)
                    te = min(j + a.mtp, L - 1)          # teacher rows needed
                    n_rows = torch.tensor([max(0, min(j, L - 1 - o) - i)
                                           for o in range(a.mtp + 1)])
                    terms = checkpoint(_chunk_mtp_terms, h_s[i:j], z[i:j],
                                       h_t[i:te], W_s, W_t, nxt[i:te], n_rows,
                                       a.divergence, _taid_lam, a.ce_beta,
                                       a.ce_mix, a.taid_space, a.rev_weight,
                                       use_reentrant=False)
                    tot = tot + terms[:, 0].sum(-1)
                    n += j - i
                    m_tot = m_tot + (terms[0, 1:] + a.ce_beta * terms[1, 1:]).sum()
                    m_n += int(n_rows[1:].sum())
                mtp_loss = m_tot / max(m_n, 1)
                del z
            else:
                for i in range(op_lo, L - 1, a.fullkl_chunk):
                    j = min(i + a.fullkl_chunk, L - 1)
                    terms = checkpoint(_chunk_div_terms, h_s[i:j], h_t[i:j],
                                       W_s, W_t, a.divergence, _taid_lam,
                                       nxt[i:j], a.ce_beta, a.ce_mix, a.taid_space,
                                       a.rev_weight, use_reentrant=False)
                    tot = tot + terms.sum(-1)
                    n += j - i
            div_term, data_term = tot[0] / max(n, 1), tot[1] / max(n, 1)
            loss = div_term + a.ce_beta * data_term
            hist["div"].append(div_term.item())
            hist["data"].append(data_term.item())
            hist["taid_t"].append(_taid_lam)
            if taid is not None:
                taid.update(step, div_term.item())
            if a.mtp:
                loss = loss + a.mtp_weight * mtp_loss
                hist.setdefault("mtp", []).append(mtp_loss.item())
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
        if a.relora_every and a.relora_warmup:
            since = step - last_merge
            if 0 <= since < a.relora_warmup:
                scale = max(since, 1) / a.relora_warmup
                for g in opt.param_groups:
                    g["lr"] *= scale
        if pull_ref:
            lr_now = sched.get_last_lr()[0]
            with torch.no_grad():
                for prm, ref in pull_ref:
                    prm.data.lerp_(ref, lr_now * a.pull_to_init)
        if a.relora_every and step > 0 and step % a.relora_every == 0:
            nm = merge_and_restart(student, opt)
            # VeRA needs its own merge, and crucially a NEW random A/B: its
            # constraint is the fixed subspace, not the parameter count, so
            # merging without re-drawing accumulates nothing.
            nm += merge_vera(student, optimizer=opt, reseed=True,
                             seed=1000 + step, d_init=a.vera_d_init,
                             verbose=False)
            last_merge = step
            print(f'  [relora] merged and restarted {nm} adapters at step '
                  f'{step} (merge {step//a.relora_every})', flush=True)
        if a.state_passing:
            promote_state(student)   # after backward, so recompute stays valid
        hist["loss"].append(loss.item())
        # Resume snapshots belong on their OWN cadence. Writing them only inside
        # evaluate() tied recoverability to --eval-every, whose default is 150 --
        # so a 150-step run wrote exactly one resume file, at the end, and an
        # interruption at step 149 lost everything. An eval is two full 8192
        # forwards; this is a 1.1 GiB overwrite of a file that already exists.
        if (a.resume_every and step % a.resume_every == 0 and _resume_ctx
                and step > 0):
            try:
                _stv = os.statvfs("/")
                if _stv.f_bavail * _stv.f_frsize > 8 * 2**30:
                    save_resume(str(CKPT_DIR / f"resume-{a.tag}.pt"),
                                student, _resume_ctx["opt"], _resume_ctx["sched"],
                                step, _resume_ctx["gens"], a.save_merged_bases,
                                _resume_ctx["taid"])
            except Exception as _e:
                print(f"  (resume save failed at step {step}: {_e})", flush=True)
        del t_logits, s_logits
        if step % a.log_every == 0:
            el = time.perf_counter() - t0
            tok_s = seen / el
            extra = (f"  div {hist['div'][-1]:.4f}  data(excess CE) "
                     f"{hist['data'][-1]:+.4f}  taid t {_taid_lam:.3f}"
                     if hist["div"] else "")
            if hist.get("mtp"):
                extra += f"  mtp {hist['mtp'][-1]:.4f}"
            _g, _a = guard.temps()
            print(f"  step {step:>4}/{a.steps}  loss {loss.item():8.4f}{extra}  "
                  f"{tok_s:6.1f} tok/s  {el/60:5.1f} min  | paced "
                  f"{pacer.paused_s / max(el, 1e-9):4.0%} ({pacer.n_pauses})  "
                  f"{pacer.power_w():4.0f} W  GPU {_g:.0f}C "
                  f"ACPI {_a:.0f}C  avail {guard.mem_available_gb():.0f} GiB  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
                  flush=True)
        if step % a.eval_every == 0:
            evaluate(step)
        torch.cuda.empty_cache()
        if a.cooldown:
            time.sleep(a.cooldown)

    evaluate(a.steps)
    out = str(LOGS_DIR / f"recovery-{a.tag}.json")
    json.dump({"args": vars(a), **hist}, open(out, "w"), indent=2)
    # save the trained parameters -- without this a run's weights are lost and
    # only the eval curve survives.
    adp = str(CKPT_DIR / f"adapters-{a.tag}.pt")
    n_saved = save_trainable(student, adp, a.save_merged_bases)
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
    if a.on_policy > 0.0:
        print(f"  on-policy: {_op_steps} rollouts; {_op_inelig} windows had no "
              f"restatement header (natural text, nothing to retrieve) and "
              f"{_op_skipped} eligible windows lost the draw")
    if _best["step"] >= 0:
        fin = hist["eval"][-1][8192]["ppl"]
        gap = (fin / _best["ppl"] - 1) * 100
        print(f"  BEST ppl@8192 {_best['ppl']:.3f} at step {_best['step']} "
              f"(final {fin:.3f}, {gap:+.2f}% worse) -> "
              f"ckpt/adapters-{a.tag}-best.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
