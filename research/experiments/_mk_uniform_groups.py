"""Uniform MLA allocation at the SAME total budget as the heterogeneous plan.

Isolates allocation POLICY from budget. The heterogeneous plan water-fills ranks
by spectra/retrieval score, which minimises reconstruction error AT INIT -- a
proxy computed with zero training, on the PRE-conversion model. Two reasons that
may not be the right objective:

  * the rank is a permanent bottleneck. A layer given 188 of 2048 possible dims
    can never represent more than 188 dims of KV subspace however long it trains.
  * the statistics describe a function that no longer exists. GDN-2 replaced 24
    layers with linear attention and NoPE removed positional encoding, so what
    each surviving softmax layer has to do changed after the covariances were
    measured.

Measured spread in the heterogeneous plan: layer 7 gets 9% of its joint maximum
while layer 19 gets 39%, a 4x difference in how hard layers are squeezed.
"""
import json

SRC = "cache/mla_groups_retr_4096_mix.json"
DST = "cache/mla_groups_uniform_4096.json"
HEAD_DIM = 256

src = json.load(open(SRC))["groups"]
layers = sorted(src, key=int)
per = 4096 // len(layers)
out = {k: [[[0, 1, 2, 3], per]] for k in layers}
json.dump({"groups": out}, open(DST, "w"), indent=1)

print(f"uniform: {len(layers)} layers x {per} = {per * len(layers)} total "
      f"(heterogeneous total was 4096)")
print("\nheterogeneous plan, each group as % of its joint maximum:")
for k in layers:
    for h, r in src[k]:
        jm = 2 * len(h) * HEAD_DIM
        print(f"  layer {k:>2}  heads {str(h):12s} {r:4d}/{jm:<4d} = {100 * r / jm:4.1f}%")
print(f"\nuniform plan: every layer {per}/2048 = {100 * per / 2048:.1f}%")
print(f"wrote {DST}")
