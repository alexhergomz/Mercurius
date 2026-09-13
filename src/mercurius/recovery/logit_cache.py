"""Offline top-K teacher logits + top-K truncated KL.

Two problems solved at once.

COMPUTE. We currently run a full teacher forward every step of every run --
roughly a third of per-step cost -- and re-pay it for every configuration in the
experiment matrix. Caching the teacher's top-K once makes it a one-time cost.
(Efficient Knowledge Distillation for LLMs: Offline Top-K Logits, 2608.03796.)

LOSS COST. The vocabulary is 248,320. A full-vocab softmax per token is the
single most expensive tensor op in the objective, and it is what made a naive
.float() allocate 32.5 GiB at 32k positions. Restricting KL to the teacher's
top-K support turns it into "a correction objective defined on a teacher-selected
domain rather than full vocabulary" -- k=64 instead of 248,320.

Storage: k=64 at fp16 values + int32 indices = 384 B/token. 5 M tokens = 1.9 GB.
"""
import os, torch
import torch.nn.functional as F


def build_cache(teacher, ids, out_path, k=64, seq=2048, n_tokens=None,
                device="cuda", verbose=True, floor_gb=8.0, chunk=2048):
    """One teacher pass; store top-K logit values and indices per position.

    The cache is keyed by position in `ids`, so training must sample windows
    from the same token array. Returns the number of tokens cached.

    n_tokens is REQUIRED. It previously defaulted to the whole corpus, which on
    a 26.3M-token corpus means a 10 GB allocation in RAM and a 10 GB write --
    on a Jetson whose root filesystem, if filled to zero, can be corrupted
    badly enough to need a reflash. A convenience default is not worth that.
    """
    import shutil
    if n_tokens is None:
        raise ValueError(
            f"n_tokens is required. The corpus holds {len(ids):,} tokens; "
            f"caching all of them at k={k} would need "
            f"{len(ids) * k * 6 / 2**30:.1f} GiB of RAM and the same on disk.")
    n_tokens = min(int(n_tokens), len(ids) - 1)
    n_tokens = (n_tokens // seq) * seq
    if n_tokens <= 0:
        raise ValueError("n_tokens smaller than one sequence")

    need_gb = n_tokens * k * 6 / 2**30
    free_gb = shutil.disk_usage("/").free / 2**30
    if free_gb - need_gb < floor_gb:
        raise SystemExit(
            f"REFUSING: cache needs {need_gb:.1f} GiB, {free_gb:.1f} GiB free, "
            f"floor {floor_gb:.0f} GiB. Lower --cache-tokens or free space.")
    if verbose:
        print(f"  cache: {n_tokens:,} tokens x k={k} -> {need_gb:.2f} GiB "
              f"({free_gb:.1f} GiB free)", flush=True)

    vals = torch.empty(n_tokens, k, dtype=torch.float16)
    idxs = torch.empty(n_tokens, k, dtype=torch.int32)
    # Full-vocabulary logsumexp, +4 B/token (~1% cache growth, zero extra FLOPs
    # since the logits are already formed). Without it the retained top-k mass
    # p(S) = sum_j exp(v_j - logZ) is not computable, so we cannot tell how often
    # the true next token falls OUTSIDE the top-64 -- positions where the loss is
    # simply silent. Published result: top-32 can retain 99.99% of probability
    # mass while containing the decision-critical token only 0.4% of the time,
    # so high retained mass is not evidence of adequate support.
    lse = torch.empty(n_tokens, dtype=torch.float32)

    import time
    from mercurius.surgery.norm_fusion import get_trunk
    teacher.eval()
    done = 0
    _t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_tokens, seq):
            x = ids[start:start + seq].unsqueeze(0).to(device)
            # CHUNKED HEAD. teacher(x).logits materializes (seq, 248320): 15.16
            # GiB bf16 and 30.3 GiB after .float() at seq=32768, which is the
            # only reason this function was ever limited to seq=2048. Running
            # the trunk once and applying lm_head over position chunks keeps the
            # peak at ~1.9 GiB, so the teacher can finally be given a context as
            # long as the student's.
            #
            # That limit was not cosmetic: with seq=2048 the cached teacher
            # distribution at any position saw <=2047 tokens (mean ~1024) while
            # training draws windows up to 32768. Forward KL is mode-covering,
            # so a less-informed, broader teacher actively pushes the student
            # FLATTER -- penalising it for retrieving a needle the teacher could
            # not see. The objective fought the metric.
            h = get_trunk(teacher)(input_ids=x).last_hidden_state[0]
            W = teacher.lm_head.weight
            for c0 in range(0, h.shape[0], chunk):
                c1 = min(c0 + chunk, h.shape[0])
                lg = (h[c0:c1] @ W.T).float()
                v, i = lg.topk(k, dim=-1)
                vals[start + c0:start + c1] = v.half().cpu()
                idxs[start + c0:start + c1] = i.int().cpu()
                lse[start + c0:start + c1] = torch.logsumexp(lg, dim=-1).cpu()
                del lg, v, i
            del h
            done += seq
            # Report every 5 blocks with a rate and an ETA. At seq=32768 the old
            # 50-block cadence meant 1.6M tokens between prints -- long enough
            # that a run could not be distinguished from a hung one, or a 3-hour
            # job from a 10-hour one, without waiting an hour to find out.
            nb = start // seq + 1
            if verbose and (nb % 5 == 0 or nb == 1):
                el = time.perf_counter() - _t0
                rate = done / max(el, 1e-9)
                eta = (n_tokens - done) / max(rate, 1e-9) / 60
                print(f"    cached {done:,}/{n_tokens:,} tokens  "
                      f"block {nb}/{n_tokens // seq}  {rate:.0f} tok/s  "
                      f"ETA {eta:.0f} min", flush=True)
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"k": k, "seq": seq, "n_tokens": n_tokens,
                "vals": vals, "idxs": idxs, "lse": lse}, out_path)
    if verbose:
        mb = (vals.numel() * 2 + idxs.numel() * 4) / 1e6
        print(f"  wrote {out_path}  ({n_tokens:,} tokens, k={k}, {mb:.0f} MB)")
    return n_tokens


def load_cache(path, device="cuda"):
    c = torch.load(path, map_location="cpu")
    return c


def topk_kl_terms(s_sel, t_vals, temperature=1.0):
    """Per-position KL, from PRE-GATHERED student logits. s_sel/t_vals (T, k).

    Split out of topk_kl so the training loop can supply the k selected columns
    directly instead of a (T, 248320) tensor it would immediately throw away.
    Returns per-position terms rather than a mean so a caller chunking over
    positions can sum and divide once -- averaging per chunk and then averaging
    the chunks would silently mis-weight a short final chunk.
    """
    s_lp = F.log_softmax(s_sel.float() / temperature, -1)
    t_lp = F.log_softmax(t_vals.float() / temperature, -1)
    p_t = t_lp.exp()
    return (p_t * (t_lp - s_lp)).sum(-1)


def topk_kl(student_logits, t_vals, t_idxs, temperature=1.0):
    """KL(teacher || student) restricted to the teacher's top-K support.

    Both distributions are renormalized over the SAME k indices, so no
    full-vocabulary softmax is ever formed. student_logits (T, V);
    t_vals/t_idxs (T, k).

    Delegates to topk_kl_terms so this and the chunked training path cannot
    drift apart -- they are the same arithmetic by construction.
    """
    s_sel = student_logits.gather(-1, t_idxs.long())        # (T, k)
    return topk_kl_terms(s_sel, t_vals, temperature).mean()


def taid_kl(student_logits, t_vals, t_idxs, lam, temperature=1.0):
    """TAID (2501.16937): KL to a time-dependent interpolation between the
    student's own distribution and the teacher's, rather than to the teacher
    directly. lam goes 0 -> 1 over training, so the target starts near the
    student and migrates to the teacher, which is what avoids the capacity-gap
    and mode-collapse failure modes of distilling straight to a stronger model.
    """
    s_sel = student_logits.gather(-1, t_idxs.long())
    return taid_kl_terms(s_sel, t_vals, lam, temperature).mean()


def taid_kl_terms(s_sel, t_vals, lam, temperature=1.0):
    """Per-position TAID terms, from PRE-GATHERED student logits. See
    topk_kl_terms for why this returns terms rather than a mean."""
    s_lp = F.log_softmax(s_sel.float() / temperature, -1)
    t_lp = F.log_softmax(t_vals.float() / temperature, -1)
    # interpolate in probability space, then renormalize
    p_mix = (1.0 - lam) * s_lp.exp().detach() + lam * t_lp.exp()
    p_mix = p_mix / p_mix.sum(-1, keepdim=True).clamp_min(1e-9)
    return (p_mix * (p_mix.clamp_min(1e-9).log() - s_lp)).sum(-1)
