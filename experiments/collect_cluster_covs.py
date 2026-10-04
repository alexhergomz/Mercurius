"""Per-cluster input covariances for Mixture-of-Latents initialisation.

WHY. #43 measured that per-cluster whitened SVD halves key reconstruction error at
E=4/r=512 (-41%) and reaches parity with plain r=512 at E=8/r=256, i.e. HALF the
cache. The construction is exactly CARE's -- whiten by a covariance, SVD, keep top-r
-- but with ONE COVARIANCE PER CLUSTER instead of one globally. This script produces
those covariances.

WHAT IT IS FOR, precisely (#42): initialisation ONLY. The router that ships is an
ordinary learned MoE router and the expert weights train; nothing stays pinned to
these clusters. Two consequences:
  * the reconstruction gains in #43 may wash out, as the allocation advantage in #25
    did (92% recovered by training). #29.3 is the counter-example where a
    factorisation-quality difference did NOT wash out, and per-cluster is that kind.
  * the DURABLE justification is symmetry breaking (#42.2): with identical-copy
    experts the pairwise cosine between expert GRADIENTS is 0.889, so the mechanism
    may never differentiate in 150 steps. A differentiated init is what makes MoL
    trainable at all, and that reason cannot wash out.

WHY TWO PASSES. k-means needs only a subsample, but a 2560x2560 covariance needs far
more than 2560 vectors per cluster to be usable, so the clusters are found on a
subsample and the covariances then accumulated over every token.

  pass 1   reservoir-subsample the hidden states entering k_proj, per layer, and
           k-means them into E clusters
  pass 2   assign every token to its nearest centroid and accumulate E covariances

Output, per attention layer: covs (E, d, d), centroids (E, d), counts (E,). The
centroids also initialise the router, so it starts matched to the experts instead of
random -- which is the whole point of the exercise.

    .venv/bin/python experiments/collect_cluster_covs.py --experts 4
    .venv/bin/python experiments/collect_cluster_covs.py --experts 8 --subsample 60000
"""
import argparse
import torch

from mercurius.calibration.care import build
from mercurius.surgery.norm_fusion import get_trunk


def _kpp(X, E, gen):
    """k-means++ seeding. Plain random seeding FAILED a unit test here: on three
    equal well-separated blobs it merged two and split one (sizes 173/800/227,
    centre error 11.4), a textbook bad-init local minimum. At E=4-8 on 2560-dim
    hidden states that would silently corrupt the whole initialisation, so the
    seeding is not optional."""
    C = [X[torch.randint(X.shape[0], (1,), generator=gen, device=X.device)][0]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(E - 1):
        p = d2.clamp_min(0)
        p = p / p.sum() if float(p.sum()) > 0 else torch.ones_like(p) / p.numel()
        C.append(X[int(torch.multinomial(p, 1, generator=gen))])
        d2 = torch.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    return torch.stack([c.reshape(-1) for c in C])


def _lloyd(X, C, iters):
    for it in range(iters):
        a = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
        for s in range(0, X.shape[0], 4096):
            blk = X[s:s + 4096]
            a[s:s + 4096] = (blk @ C.T * -2 + (C * C).sum(1)).argmin(1)
        newC = C.clone()
        for e in range(C.shape[0]):
            m = a == e
            if m.any():
                newC[e] = X[m].mean(0)
        if torch.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC
    inertia = float(((X - C[a]) ** 2).sum())
    return C, a, inertia


def _kmeans(X, E, iters=50, seed=0, restarts=4, verbose=True):
    """k-means++ with restarts, best inertia wins."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    best = None
    for rs in range(restarts):
        C, a, inertia = _lloyd(X, _kpp(X, E, g), iters)
        if best is None or inertia < best[2]:
            best = (C, a, inertia)
    C, a, inertia = best
    if verbose:
        cnt = torch.bincount(a, minlength=E).tolist()
        print(f"      k-means++ x{restarts}: inertia {inertia:.4g}, sizes {cnt}",
              flush=True)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--samples", type=int, default=256,
                    help="calibration sequences per pass")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--subsample", type=int, default=40000,
                    help="hidden vectors per layer kept for k-means. Needs to be "
                         "comfortably above E*d for the per-cluster covariances to "
                         "be worth whitening with; d is 2560 here.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    E = a.experts
    out = a.out or f"cache/kv_covs_cluster_E{E}.pt"

    m, calib = build()
    trunk = get_trunk(m)
    idx = [i for i, l in enumerate(trunk.layers) if hasattr(l, "self_attn")]
    d = trunk.layers[0].input_layernorm.weight.shape[0]
    print(f"  {len(idx)} attention layers, d_model {d}, E={E}", flush=True)

    # ---------------- pass 1: subsample + k-means ----------------
    keep = {i: [] for i in idx}
    kept = {i: 0 for i in idx}
    g = torch.Generator().manual_seed(a.seed)

    def mk1(i):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            if kept[i] < a.subsample:                  # simple prefix subsample
                take = min(a.subsample - kept[i], x.shape[0])
                keep[i].append(x[:take].float().cpu())
                kept[i] += take
        return hook

    hooks = [trunk.layers[i].self_attn.k_proj.register_forward_hook(mk1(i))
             for i in idx]
    with torch.no_grad():
        for s in range(a.samples):
            off = int(torch.randint(0, len(calib) - a.seq - 1, (1,), generator=g))
            m(input_ids=calib[off:off + a.seq].unsqueeze(0).cuda(), logits_to_keep=1)
            if (s + 1) % 64 == 0:
                print(f"    pass 1: {s+1}/{a.samples} "
                      f"({kept[idx[0]]:,} vectors kept)", flush=True)
            if all(v >= a.subsample for v in kept.values()):
                print(f"    pass 1: subsample full at sequence {s+1}", flush=True)
                break
    for h in hooks:
        h.remove()

    cents = {}
    for i in idx:
        X = torch.cat(keep[i]).cuda()
        print(f"    layer {i}: k-means over {X.shape[0]:,} vectors", flush=True)
        cents[i] = _kmeans(X, E, seed=a.seed).cpu()
        del X
        torch.cuda.empty_cache()
    keep.clear()

    # ---------------- pass 2: per-cluster covariance ----------------
    covs = {i: torch.zeros(E, d, d, dtype=torch.float64, device="cuda") for i in idx}
    cnts = {i: torch.zeros(E, dtype=torch.long) for i in idx}
    cg = {i: cents[i].cuda() for i in idx}

    def mk2(i):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            C = cg[i]
            asg = (x @ C.T * -2 + (C * C).sum(1)).argmin(1)
            for e in range(E):
                sel = x[asg == e]
                if sel.numel():
                    covs[i][e] += sel.double().T @ sel.double()
                    cnts[i][e] += sel.shape[0]
        return hook

    hooks = [trunk.layers[i].self_attn.k_proj.register_forward_hook(mk2(i))
             for i in idx]
    g = torch.Generator().manual_seed(a.seed + 1)
    with torch.no_grad():
        for s in range(a.samples):
            off = int(torch.randint(0, len(calib) - a.seq - 1, (1,), generator=g))
            m(input_ids=calib[off:off + a.seq].unsqueeze(0).cuda(), logits_to_keep=1)
            if (s + 1) % 64 == 0:
                print(f"    pass 2: {s+1}/{a.samples}", flush=True)
            torch.cuda.empty_cache()
    for h in hooks:
        h.remove()

    payload = {}
    for i in idx:
        c = cnts[i].clamp_min(1).to(torch.float64)
        payload[i] = {"covs": (covs[i] / c[:, None, None]).float().cpu(),
                      "centroids": cents[i],
                      "counts": cnts[i]}
        frac = (cnts[i].float() / cnts[i].sum()).tolist()
        thin = [j for j, f in enumerate(frac) if f * E < 0.25]
        note = f"  !! clusters {thin} hold <1/4 of their even share" if thin else ""
        print(f"    layer {i}: counts {cnts[i].tolist()} "
              f"({', '.join(f'{f:.2f}' for f in frac)}){note}", flush=True)
    torch.save({"experts": E, "layers": payload}, out)
    print(f"  wrote {out}", flush=True)
    print("  NOTE these are an INITIALISATION (#42). The shipped router is learned "
          "and may abandon this partition entirely.", flush=True)


if __name__ == "__main__":
    main()
