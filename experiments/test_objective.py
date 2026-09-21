"""Numerical checks of the distillation objective's algebra.

Every claim in recovery/train.py's _chunk_div_terms / taid_space_for docstrings
is asserted here in float64 with autograd, on random logits with a
realistically peaked teacher. Gradients are taken w.r.t. the STUDENT LOGITS,
which is what the network sees.

    python experiments/test_objective.py
"""
import math
import torch
import torch.nn.functional as F

from mercurius.recovery.train import _chunk_div_terms, TAIDSchedule

torch.manual_seed(0)
V, C = 2048, 16
DT = torch.float64


def grad_of(fn, z):
    z = z.clone().requires_grad_(True)
    fn(z).sum().backward()
    return z.grad


def fwd(t_lp, s_lp):   # KL(t || s)
    return (t_lp.exp() * (t_lp - s_lp)).sum(-1)


def rev(t_lp, s_lp):   # KL(s || t)
    return (s_lp.exp() * (s_lp - t_lp)).sum(-1)


def target(z_s, z_t, lam, space):
    s_lp = F.log_softmax(z_s.detach(), -1)
    t_lp = F.log_softmax(z_t, -1)
    if space == "logit":
        return F.log_softmax((1 - lam) * s_lp + lam * t_lp, -1)
    return ((1 - lam) * s_lp.exp() + lam * t_lp.exp()).log()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


z_t = 3.0 * torch.randn(C, V, dtype=DT)          # peaked teacher
z_s = z_t + 1.5 * torch.randn(C, V, dtype=DT)    # a student that disagrees
checks = []


def check(name, ok, detail):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


print("1. degenerate TAID pairings (exact loss-scale ramps)")
for lam in (0.1, 0.4, 0.9):
    g = grad_of(lambda z: fwd(target(z_s, z_t, lam, "prob"), F.log_softmax(z, -1)), z_s)
    g_ref = lam * grad_of(lambda z: fwd(F.log_softmax(z_t, -1), F.log_softmax(z, -1)), z_s)
    check(f"prob-space + forward == {lam} * forward KL", rel(g, g_ref) < 1e-10,
          f"relL2 {rel(g, g_ref):.1e}")
    g = grad_of(lambda z: rev(target(z_s, z_t, lam, "logit"), F.log_softmax(z, -1)), z_s)
    g_ref = lam * grad_of(lambda z: rev(F.log_softmax(z_t, -1), F.log_softmax(z, -1)), z_s)
    check(f"logit-space + reverse == {lam} * reverse KL", rel(g, g_ref) < 1e-10,
          f"relL2 {rel(g, g_ref):.1e}")

print("2. the genuine pairings are NOT scalings, and have the stated limits")
for space, D, lim, lim_name in (("logit", fwd, rev, "reverse"),
                                ("prob", rev, fwd, "forward")):
    lam = 0.4
    g = grad_of(lambda z: D(target(z_s, z_t, lam, space), F.log_softmax(z, -1)), z_s)
    g_full = grad_of(lambda z: D(F.log_softmax(z_t, -1), F.log_softmax(z, -1)), z_s)
    cos = F.cosine_similarity(g.flatten(), g_full.flatten(), 0).item()
    check(f"{space}-space + {D.__name__} at t=0.4 is not parallel to t=1",
          cos < 0.999, f"cosine {cos:.4f}")
    lam = 1e-4
    g = grad_of(lambda z: D(target(z_s, z_t, lam, space), F.log_softmax(z, -1)), z_s)
    g_lim = lam * grad_of(lambda z: lim(F.log_softmax(z_t, -1), F.log_softmax(z, -1)), z_s)
    check(f"{space}-space + {D.__name__} at t->0 ~ t * {lim_name} KL",
          rel(g, g_lim) < 1e-2, f"relL2 {rel(g, g_lim):.1e}")

print("3. the data term")
h_t = torch.randn(C, 64, dtype=DT)
W = torch.randn(V, 64, dtype=DT)
y = torch.randint(0, V, (C,))
out = _chunk_div_terms(h_t, h_t, W, W, "reverse", 1.0, y, 1.0)
check("student == teacher: divergence and excess CE are exactly 0",
      out.abs().max().item() == 0.0, f"max |term| {out.abs().max().item():.1e}")
h_s = h_t + 0.3 * torch.randn_like(h_t)
out = _chunk_div_terms(h_s, h_t, W, W, "reverse", 1.0, y, 1.0)
ce = F.cross_entropy(h_s @ W.T, y, reduction="none") - \
     F.cross_entropy(h_t @ W.T, y, reduction="none")
# the function computes in fp32 by design (a 248k-way softmax), hence 1e-5
check("row 1 == CE(student) - CE(teacher)", rel(out[1], ce) < 1e-5,
      f"relL2 {rel(out[1], ce):.1e}")
check("row 0 == KL(student || teacher)",
      rel(out[0], rev(F.log_softmax(h_t @ W.T, -1), F.log_softmax(h_s @ W.T, -1))) < 1e-5, "")

# Direct minimisation over a free distribution, one position: where does each
# data term drive the student? The teacher is peaked with entropy ~ half of
# log V, as a real LM's is.
z_t1 = 3.0 * torch.randn(V, dtype=DT)
t_lp = F.log_softmax(z_t1, -1)
yy = int(t_lp.argsort(descending=True)[4])   # a plausible, non-argmax token


def minimise(loss_fn, steps=20000):
    z = z_t1.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(F.log_softmax(z, -1)).backward()
        opt.step()
    s = F.log_softmax(z.detach(), -1)
    return -(s.exp() * s).sum().item(), s[yy].exp().item()


eps = 1e-4
y_lp = torch.full((V,), math.log(eps / V), dtype=DT)
y_lp[yy] = math.log(1 - eps + eps / V)
H_t = -(t_lp.exp() * t_lp).sum().item()
H_old, p_old = minimise(lambda s: rev(t_lp, s) + (rev(y_lp, s) - rev(y_lp, t_lp)))
H_new, p_new = minimise(lambda s: rev(t_lp, s) + (-s[yy] + t_lp[yy]))
H_fwd, p_fwd = minimise(lambda s: fwd(t_lp, s) + (-s[yy] + t_lp[yy]))
print(f"     teacher: H {H_t:.3f} nats, p(y) {t_lp[yy].exp().item():.4f}")
# Adam crawls on the flat softmax tail, so this is still falling at 20k steps;
# the claim is the direction and the endpoint, not the exact zero
check("OLD reverse data term collapses toward a point mass",
      H_old < 0.1 * H_t and p_old > 0.95,
      f"H {H_old:.4f} nats, p(y) {p_old:.4f}")
check("NEW reverse KL + excess CE keeps a real distribution", H_new > 0.25 * H_t,
      f"H {H_new:.3f} nats, p(y) {p_new:.4f}")
# forward: KL(t||s) + CE_y is CE against (t + delta_y)/2, so the optimum is
# exactly that mixture -- the same thing --ce-mix 0.5 targets
mix = 0.5 * (t_lp.exp() + F.one_hot(torch.tensor(yy), V).to(DT))
check("forward KL + excess CE optimum == (t + onehot)/2",
      abs(p_fwd - mix[yy].item()) < 1e-3, f"p(y) {p_fwd:.4f} vs {mix[yy].item():.4f}")

print("4. TAID schedule")
sch = TAIDSchedule(100)
ts = [sch.update(k, 1.0 / (k + 1)) for k in range(1, 101)]
check("monotone, starts >= 0.4, ends at 1.0",
      all(b >= a for a, b in zip(ts, ts[1:])) and ts[0] >= 0.4 and ts[-1] == 1.0,
      f"t: {ts[0]:.3f} {ts[49]:.3f} {ts[-1]:.3f}")

print(f"\n{sum(checks)}/{len(checks)} checks passed")
raise SystemExit(0 if all(checks) else 1)
