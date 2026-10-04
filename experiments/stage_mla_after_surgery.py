"""Re-measure the MLA statistics AFTER GDN-2 + NoPE, then allocate.

THE BUG THIS FIXES. cache/kv_covs_4b_mix.pt was collected on the model BEFORE the
surgery. The rank plan derived from it (cache/mla_groups_retr_4096_mix.json) is
therefore optimal for a function that exists at no point in the pipeline:

  * NoPE removes rotary position entirely, so the K vectors change character.
    Their covariance spectrum has no reason to resemble the pre-NoPE one.
  * the GDN-2 lift replaces 24 of 32 layers with linear attention, changing what
    the 8 surviving softmax layers receive.

Conversion ORDER was already right -- build() does GDN-2, then the dial, then MLA.
Only the statistics were stale. So the fix is narrow: collect the covariances on a
model that already has GDN-2 and NoPE applied, and allocate from those.

WHY NOT JUST MEASURE THE RANK EACH LAYER NEEDS. Two candidate measurements were
considered and both have the same defect:

  * spectral decay of the PRE-conversion KV covariance (what is used today):
    measures the rank needed to reproduce a function we are about to replace.
  * truncating a generously-budgeted TRAINED model and reading the damage:
    measures what removing capacity the model learned to use costs -- not what a
    model TRAINED at that rank could reach. A model given 200 dims from the start
    adapts around the constraint; one given 789 and cut to 200 does not. So it
    over-estimates the need, in the same way the spectral method does at the other
    end of training.

Both measure reconstruction of an existing function rather than achievable quality
under a constraint. Re-measuring after surgery does not solve that in general, but
it removes the part of the mis-specification that is simply an ordering mistake.

    python experiments/stage_mla_after_surgery.py --budget 4096
"""
import argparse
import json
import sys

import torch

sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4096,
                    help="total latent budget, matched to the existing plan so "
                         "only the STATISTICS differ")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--min-r", type=int, default=64)
    ap.add_argument("--out-covs", default="cache/kv_covs_postsurgery.pt")
    ap.add_argument("--out-groups", default="cache/mla_groups_postsurgery_4096.json")
    ap.add_argument("--reference",
                    default="cache/mla_groups_retr_4096_mix.json",
                    help="existing plan, used only to reuse its HEAD GROUPING so "
                         "the comparison isolates the statistics")
    a = ap.parse_args()

    from mercurius.models.kda import load_kda_model
    from mercurius.surgery.rope_dial import install_rope_dial
    from mercurius.surgery.kda_lift import lift_to_gdn2
    from mercurius.surgery.transmla import whitened_spectra, allocate_ranks
    from mercurius.calibration.attention import collect_attn_covariances
    from mercurius.recovery.train import CKPT, EVAL_DATA
    from mercurius.paths import STAGE_AB
    from transformers import AutoTokenizer

    print("building the model with GDN-2 + NoPE and NO MLA", flush=True)
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    trunk = m.model.language_model if hasattr(m.model, "language_model") else m.model
    for l in trunk.layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")          # NoPE, same as the recipe
    n_lift = lift_to_gdn2(m)
    print(f"  GDN-2 lift on {n_lift} layers, NoPE applied, MLA NOT applied",
          flush=True)
    m.eval()

    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    ids = torch.tensor(tok(open(EVAL_DATA).read(),
                           add_special_tokens=False).input_ids)
    print(f"calibration tokens: {len(ids):,}", flush=True)

    covs = collect_attn_covariances(m, ids, a.samples, a.seq)
    torch.save(covs, a.out_covs)
    print(f"wrote {a.out_covs}", flush=True)

    spec = whitened_spectra(m, covs)
    ranks = allocate_ranks(spec, a.budget, min_r=a.min_r)
    print(f"\nallocated {sum(ranks.values())} of budget {a.budget} "
          f"across {len(ranks)} layers", flush=True)

    ref = json.load(open(a.reference))["groups"]
    print("\nlayer   OLD (pre-surgery stats)      NEW (post-surgery stats)")
    out = {}
    for k in sorted(ref, key=int):
        old = [r for _, r in ref[k]]
        new_total = ranks.get(int(k))
        # reuse the head GROUPING, split the new total in the old proportion, so
        # only the statistics differ between plans
        if new_total is None:
            out[k] = ref[k]
            print(f"  {k:>3}   {str(old):26s} (unchanged, no new spectrum)")
            continue
        tot_old = sum(old)
        split = [max(a.min_r, round(new_total * r / tot_old)) for r in old]
        out[k] = [[heads, s] for (heads, _), s in zip(ref[k], split)]
        print(f"  {k:>3}   {str(old):26s} {split}")
    json.dump({"groups": out}, open(a.out_groups, "w"), indent=1)
    tot = sum(r for v in out.values() for _, r in v)
    print(f"\nnew plan total {tot} (reference was "
          f"{sum(r for v in ref.values() for _, r in v)})")
    print(f"wrote {a.out_groups}")


if __name__ == "__main__":
    main()
