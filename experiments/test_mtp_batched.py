"""The batched MTP loss must equal the per-head loop it replaced, exactly.

Reference: the original trainer code -- the main loss chunk by chunk, then for
each head j its own chunked loop, recomputing the teacher's logits at p+j.
Batched: _chunk_mtp_terms (teacher log-softmax once per chunk, the student's
K+1 row sets in one GEMM). Compared on losses and on gradients into the
student states and every head, in float64 on the CPU, with a sequence length
that is not a multiple of the chunk so the far heads lose rows at the end.

    python experiments/test_mtp_batched.py
"""
import torch
from mercurius.recovery.train import _chunk_div_terms, _chunk_mtp_terms

torch.manual_seed(0)
DT = torch.float64
V, ds, dt, L, K, C = 997, 24, 40, 301, 4, 64
W_s, W_t = torch.randn(V, ds, dtype=DT), torch.randn(V, dt, dtype=DT)
x = torch.randint(0, V, (1, L))
nxt = x[0, 1:]
ok = []
for mode, lam, space in (("reverse", 0.6, "prob"), ("forward", 0.5, "logit"),
                         ("jeffreys", 1.0, "prob")):
    h_s0 = torch.randn(L, ds, dtype=DT); z0 = torch.randn(L, K, ds, dtype=DT)
    h_t = torch.randn(L, dt, dtype=DT)
    args = (mode, lam)

    # --- reference: the per-head loops the trainer used before
    h_s = h_s0.clone().requires_grad_(); z = z0.clone().requires_grad_()
    tot, n = 0.0, 0
    for i in range(0, L - 1, C):
        j = min(i + C, L - 1)
        t = _chunk_div_terms(h_s[i:j], h_t[i:j], W_s, W_t, mode, lam, nxt[i:j], 1.0, 0.0, space, 0.7)
        tot = tot + t.sum(-1); n += j - i
    main_ref = (tot[0] + tot[1]) / n
    m_tot, m_n = 0.0, 0
    for jh in range(1, K + 1):
        hi = L - 1 - jh
        for i in range(0, hi, C):
            jj = min(i + C, hi)
            tm = _chunk_div_terms(z[i:jj, jh - 1], h_t[i + jh:jj + jh], W_s, W_t, mode, lam,
                                  x[0, i + 1 + jh:jj + 1 + jh], 1.0, 0.0, space, 0.7)
            m_tot = m_tot + (tm[0] + tm[1]).sum(); m_n += jj - i
    mtp_ref = m_tot / m_n
    (main_ref + 0.1 * mtp_ref).backward()
    g_ref = (h_s.grad.clone(), z.grad.clone())

    # --- batched
    h_s = h_s0.clone().requires_grad_(); z = z0.clone().requires_grad_()
    tot, n, m_tot, m_n = 0.0, 0, 0.0, 0
    for i in range(0, L - 1, C):
        j = min(i + C, L - 1)
        te = min(j + K, L - 1)
        n_rows = torch.tensor([max(0, min(j, L - 1 - o) - i) for o in range(K + 1)])
        t = _chunk_mtp_terms(h_s[i:j], z[i:j], h_t[i:te], W_s, W_t, nxt[i:te], n_rows,
                             mode, lam, 1.0, 0.0, space, 0.7)
        tot = tot + t[:, 0].sum(-1); n += j - i
        m_tot = m_tot + (t[0, 1:] + t[1, 1:]).sum(); m_n += int(n_rows[1:].sum())
    main_b, mtp_b = (tot[0] + tot[1]) / n, m_tot / m_n
    (main_b + 0.1 * mtp_b).backward()

    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    # relative: the per-row math runs in fp32 BY DESIGN (the .float() on the
    # logits), and the batched path sums rows in a different order, so the
    # loss can differ at fp32 rounding; gradients are per-row and must match
    e = [abs(main_b.item() - main_ref.item()) / abs(main_ref.item()),
         abs(mtp_b.item() - mtp_ref.item()) / abs(mtp_ref.item()),
         rel(h_s.grad, g_ref[0]), rel(z.grad, g_ref[1])]
    ok.append(max(e) < 1e-6 and m_n == sum(L - 1 - jh for jh in range(1, K + 1)))
    print(f"{mode:>8} lam={lam}: rel d main {e[0]:.1e} rel d mtp {e[1]:.1e} "
          f"grad h_s {e[2]:.1e} grad z {e[3]:.1e} rows {m_n}  [{'PASS' if ok[-1] else 'FAIL'}]")
print(f"{sum(ok)}/{len(ok)} passed")
raise SystemExit(0 if all(ok) else 1)
