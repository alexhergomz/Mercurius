"""How much decode speed would the conv MTP head buy, as a self-speculative draft?

Self-speculative decoding is LOSSLESS: the heads propose t+2..t+K+1, the model
verifies them in one batched forward, and a mismatch falls back to the model's
own token. So the quantity that matters is not whether a head predicts the
TEXT, but whether it predicts what THIS MODEL would have emitted. Measured
here by teacher forcing: head j at position t is accepted iff its argmax equals
the model's own argmax at position t+j (greedy decoding).

Acceptance is CHAINED -- a draft token is only used if every earlier draft in
the block was accepted -- so the expected accepted tokens per verification step
is

    tau = 1 + sum_j P(drafts 1..j all correct)

and, decode being memory-bound, the speedup is about tau scaled by what a
(tau+1)-token verification forward costs against a 1-token forward. Both are
timed here rather than assumed.

    python experiments/mtp_acceptance.py --adapters ckpt/adapters-...-best.pt
"""
import argparse
import time

import torch
from transformers import AutoTokenizer

from mercurius import guard
from mercurius.eval.retrieval_ab import build
from mercurius.paths import CACHE_DIR, CKPT_DIR, STAGE_AB, WIKITEXT
from mercurius.surgery.norm_fusion import get_trunk


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=str(CKPT_DIR / "adapters-4b27b-D150-best.pt"))
    ap.add_argument("--mla-groups", default=str(CACHE_DIR / "mla_groups_retr_4096.json"))
    ap.add_argument("--covs", default=str(CACHE_DIR / "kv_covs_4b.pt"))
    ap.add_argument("--n", type=int, default=8192, help="teacher-forced positions")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--bench-steps", type=int, default=32)
    a = ap.parse_args()
    guard.cap_cuda_memory(60)
    pacer = guard.ThermalPacer(84.0, 80.0, 90.0, 85.0)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    m = build(a.adapters, 512, a.covs, quantize=True, groups=a.mla_groups)
    if not hasattr(m, "mtp_head"):
        raise SystemExit("this checkpoint has no MTP head")
    pacer.attach(m)
    K = m.mtp_head.k
    ids = tok(open(WIKITEXT).read(), return_tensors="pt").input_ids[0][:a.n].cuda()
    x = ids.unsqueeze(0)

    h = get_trunk(m)(input_ids=x, use_cache=False).last_hidden_state[0]
    W = m.get_output_embeddings().weight
    own = torch.empty(h.shape[0], dtype=torch.long, device=h.device)
    for i in range(0, h.shape[0], a.chunk):                 # the model's own greedy token
        j = min(i + a.chunk, h.shape[0])
        own[i:j] = (h[i:j] @ W.T).argmax(-1)
    z = m.mtp_head(h.unsqueeze(0))[0]                       # (T, K, d)
    T = h.shape[0]
    acc = torch.zeros(K, dtype=torch.bool, device=h.device).unsqueeze(0).repeat(T - K - 1, 1)
    for jj in range(K):
        pred = torch.empty(T - K - 1, dtype=torch.long, device=h.device)
        for i in range(0, T - K - 1, a.chunk):
            e = min(i + a.chunk, T - K - 1)
            pred[i:e] = (z[i:e, jj] @ W.T).argmax(-1)
        acc[:, jj] = pred == own[jj + 1:T - K + jj]         # model's own token at t+1+j
    marg = acc.float().mean(0)
    chain = acc.cumprod(1).float().mean(0)                  # all drafts up to j accepted
    tau = 1.0 + float(chain.sum())
    print(f"positions {T - K - 1:,}; K={K}")
    for jj in range(K):
        print(f"  head {jj + 1} (t+{jj + 2}): marginal accept {marg[jj]:.3f}  "
              f"chained {chain[jj]:.3f}")
    print(f"expected accepted tokens per verification step  tau = {tau:.2f}")

    # timings: one token on a cache, and a (tau+1)-token verification forward
    warm = m(input_ids=x[:, :512], use_cache=True)
    cache = warm.past_key_values
    nxt = x[:, 512:513]
    for _ in range(3):
        m(input_ids=nxt, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(a.bench_steps):
        m(input_ids=nxt, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    t1 = (time.time() - t0) / a.bench_steps
    kk = max(2, int(round(tau)) + 1)
    blk = x[:, 520:520 + kk]
    for _ in range(3):
        m(input_ids=blk, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(a.bench_steps):
        m(input_ids=blk, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    tk = (time.time() - t0) / a.bench_steps
    print(f"decode: 1 token {t1 * 1e3:.1f} ms ({1 / t1:.0f} tok/s); "
          f"{kk}-token verification {tk * 1e3:.1f} ms ({tk / t1:.2f}x a single token)")
    print(f"=> self-speculative speedup ~ tau / (verify cost) = {tau / (tk / t1):.2f}x "
          f"({1 / t1:.0f} -> {tau / tk:.0f} tok/s), lossless (greedy)")
    pacer.detach()


if __name__ == "__main__":
    main()
