"""Ungrouped latents at the SAME per-layer rank as the water-filled plan.

Isolates the BLOCK-DIAGONAL RESTRICTION from the rank allocation, which
uniform150 conflated (it changed both the ranks and the grouping).

Why it should help, and why nothing has tested it. The grouped path builds
up_k/up_v block-structured: head h reads only its group's latent columns and
zeros elsewhere. That is the shared construction WITH ZEROS IMPOSED -- strictly
fewer degrees of freedom at identical cache, since the cache is the concatenated
latent either way. Sharing lets heads amortise information they have in common;
partitioning forces each group to re-encode it inside its own slice.

The zeros are nominally trainable, so the restriction could lift itself. Measured
on ablate-base150 after 150 steps it does not: off-block RMS is 0.00022-0.00023
against on-block 0.045-0.068, i.e. 0.3-0.5%. Not a learning-rate ceiling either --
the latents train at 3e-5, so lr*steps = 0.0045 was reachable and they used 5% of
it. The block solution is the CARE optimum, so there is little gradient pressure
to cross it.

Grouping's stated justification was never expressiveness, it was allocation: "a
joint SVD over every KV head spends rank on whichever heads carry the most
whitened energy, which is not the same as the heads that retrieve". And decision
#25 measured allocation as worth 0.062 nats and 0.5 sigma -- it washes out. So
grouping may be paying a permanent expressiveness cost for an init-time benefit
that adaptation erases.

PER-LAYER RANK IS PRESERVED EXACTLY so only the partition changes.
"""
import json

SRC = "cache/mla_groups_retr_4096_mix.json"
DST = "cache/mla_groups_ungrouped_4096.json"

src = json.load(open(SRC))["groups"]
out, tot_src, tot_dst = {}, 0, 0
print(f"{'layer':>6}  {'water-filled groups':<26} {'-> ungrouped'}")
for k in sorted(src, key=int):
    per_layer = sum(r for _, r in src[k])
    heads = sorted({h for hs, _ in src[k] for h in hs})
    out[k] = [[heads, per_layer]]
    tot_src += per_layer
    tot_dst += per_layer
    print(f"{k:>6}  {str([[hs, r] for hs, r in src[k]]):<26} -> [{heads}, {per_layer}]")
json.dump({"groups": out}, open(DST, "w"), indent=1)
print(f"\ntotal rank {tot_src} -> {tot_dst} (identical, as required)")
print(f"wrote {DST}")
