"""Exactness gate for the chunked top-K head.

The training loop no longer forms the (T, 248320) student logit tensor. It
gathers the lm_head rows for the teacher's top-K indices and contracts against
the hidden state instead. In exact arithmetic that is the same number; in bf16
it is a different matmul reduction order, so it has to be MEASURED equal rather
than assumed -- the same discipline every other stage here went through, and the
reason the zero-centered RMSNorm bug was caught instead of shipped.

Gates BOTH:
  * the loss value, and
  * the gradient that actually reaches the trainable parameters,
because a loss that matches while the gradient does not would train quietly
wrong, which is the worst failure mode available here.

Thresholds are relative, and generous enough for bf16 accumulation order but far
tighter than any real bug would produce.
"""
import sys, argparse, torch
from transformers import AutoTokenizer
from torch.utils.checkpoint import checkpoint

from mercurius.calibration.care import build, CKPT
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.recovery.logit_cache import load_cache, topk_kl, taid_kl
from mercurius.adapters.lora import trainable_parameters
from mercurius.recovery.train import _chunk_kl_terms
from mercurius.paths import FINEWEB

LOSS_TOL = 2e-3      # relative
GRAD_TOL = 2e-2      # relative, on the global grad vector


def grads_of(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                      .detach().float().flatten() for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096,
                    help="must be short enough that the OLD path still fits")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--head-chunk", type=int, default=2048)
    ap.add_argument("--taid", action="store_true")
    ap.add_argument("--cache", default="cache/tk64_1m.pt")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    train_ids = tok(open(str(FINEWEB)).read(),
                    return_tensors="pt").input_ids[0]
    cache = load_cache(a.cache)
    assert a.offset + a.seq <= cache["n_tokens"], "window outside cached prefix"

    x = train_ids[a.offset:a.offset + a.seq].unsqueeze(0).cuda()
    tv = cache["vals"][a.offset:a.offset + a.seq].cuda()
    ti = cache["idxs"][a.offset:a.offset + a.seq].cuda()
    lam = 0.5

    model = build()
    model.train()
    params = trainable_parameters(model)
    print(f"seq {a.seq} | k {cache['k']} | {len(params)} trainable tensors",
          flush=True)

    # ---------- OLD path: materialize every logit, then gather ----------
    for p in params:
        p.grad = None
    s_logits = model(input_ids=x).logits[0]
    peak_old = torch.cuda.max_memory_allocated() / 2**30
    loss_old = (taid_kl(s_logits, tv, ti, lam) if a.taid
                else topk_kl(s_logits, tv, ti))
    loss_old.backward()
    g_old = grads_of(params)
    del s_logits
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---------- NEW path: gather the lm_head rows, chunked ----------
    for p in params:
        p.grad = None
    h = get_trunk(model)(input_ids=x).last_hidden_state[0]
    W_lm = model.lm_head.weight
    tot, n = 0.0, 0
    for i in range(0, a.seq, a.head_chunk):
        j = min(i + a.head_chunk, a.seq)
        terms = checkpoint(_chunk_kl_terms, h[i:j], W_lm, tv[i:j], ti[i:j],
                           lam, a.taid, use_reentrant=False)
        tot = tot + terms.sum()
        n += j - i
    loss_new = tot / max(n, 1)
    loss_new.backward()
    g_new = grads_of(params)
    peak_new = torch.cuda.max_memory_allocated() / 2**30

    # ---------- verdict ----------
    lo, ln = loss_old.item(), loss_new.item()
    rel_loss = abs(ln - lo) / max(abs(lo), 1e-12)
    denom = g_old.norm().clamp_min(1e-12)
    rel_grad = ((g_new - g_old).norm() / denom).item()
    cos = torch.nn.functional.cosine_similarity(
        g_old.unsqueeze(0), g_new.unsqueeze(0)).item()

    print(f"\n  loss  old {lo:.8f}   new {ln:.8f}   rel {rel_loss:.3e}")
    print(f"  grad  ||old|| {g_old.norm():.6f}  ||new|| {g_new.norm():.6f}")
    print(f"        rel L2 {rel_grad:.3e}   cosine {cos:.8f}")
    print(f"  peak  old {peak_old:.2f} GiB   new {peak_new:.2f} GiB "
          f"({peak_old / max(peak_new, 1e-9):.2f}x less)")

    ok = rel_loss < LOSS_TOL and rel_grad < GRAD_TOL
    print(f"\n  {'PASS' if ok else 'FAIL'}: chunked head "
          f"{'matches' if ok else 'DIVERGES FROM'} the full-logit path "
          f"(tol loss {LOSS_TOL:.0e}, grad {GRAD_TOL:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
