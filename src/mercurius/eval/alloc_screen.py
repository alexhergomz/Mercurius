"""Compare d_c allocations at a fixed total KV budget, before any recovery.

Why no adapters. The trained adapters were fitted against a 256-wide latent. Re-
converting at a different allocation and then loading them mixes two effects:
the allocation, and the mismatch between the adapters and a latent they were
never trained on. Converting the stage-AB model directly and evaluating it with
no adapters at all measures only what the allocation itself costs -- how much
retrieval each split destroys. A separating result here justifies the expense of
a full training arm per allocation; a null result says the axis is not worth it.

Every arm sees the SAME samples, and the uncompressed model is included, because
a gap between two compressed models says nothing about whether either is usable.
"""
import argparse, json, os, sys, time
import torch
from transformers import AutoTokenizer
from mercurius.recovery.train import CKPT
from mercurius.paths import STAGE_AB, CACHE_DIR, LOGS_DIR
from mercurius.models.kda import load_kda_model
from mercurius.models.gdn2 import convert_to_gdn2
from mercurius.surgery.transmla import convert_to_mla
from mercurius.eval.ruler import score_sample
from mercurius.eval import ruler_gen as R


def build(stage_ab, covs_path, alloc=None, dc=None, gdn2=True):
    m = load_kda_model(stage_ab, dtype=torch.bfloat16)
    if gdn2:
        convert_to_gdn2(m)
    if alloc is not None or dc is not None:
        covs = torch.load(covs_path, map_location="cpu")
        convert_to_mla(m, d_c=(None if alloc else dc), alloc=alloc,
                       covs=covs, verbose=True)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-file",
                    default=str(LOGS_DIR / "retrieval_heads.json"))
    ap.add_argument("--keys", nargs="+", default=["uniform", "spectral", "retrieval"])
    ap.add_argument("--with-uncompressed", action="store_true", default=True)
    ap.add_argument("--stage-ab", default=str(STAGE_AB))
    ap.add_argument("--covs", default=str(CACHE_DIR / "kv_covs.pt"))
    ap.add_argument("--tasks", nargs="+", default=["niah_multivalue"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[4096])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--gen-tokens", type=int, default=-1)
    ap.add_argument("--out", default=str(LOGS_DIR / "alloc_screen.json"))
    a = ap.parse_args()

    allocs = json.load(open(a.alloc_file))
    tok = AutoTokenizer.from_pretrained(CKPT)

    print("generating samples (shared by every arm)", flush=True)
    data = {}
    for task in a.tasks:
        for n in a.lengths:
            data[(task, n)] = R.generate(task, n, a.samples, tok, verbose=False)
    print(f"  {sum(len(v) for v in data.values())} samples\n", flush=True)

    arms = []
    if a.with_uncompressed:
        arms.append(("uncompressed", None))
    for k in a.keys:
        if k not in allocs:
            print(f"  skipping {k}: not in {a.alloc_file}", flush=True)
            continue
        arms.append((k, {int(kk): int(vv) for kk, vv in allocs[k].items()}))

    rows = []
    for tag, alloc in arms:
        t0 = time.perf_counter()
        total = sum(alloc.values()) if alloc else None
        print(f"=== {tag}" + (f" (total d_c {total})" if total else " (no MLA)"),
              flush=True)
        m = build(a.stage_ab, a.covs, alloc=alloc)
        res = {}
        for task in a.tasks:
            for n in a.lengths:
                ems, nlls = [], []
                for s in data[(task, n)]:
                    try:
                        nll, nll_v, pred = score_sample(m, tok, s, a.gen_tokens)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        continue
                    nlls.append(nll_v if nll_v is not None else nll)
                    gold = s["outputs"] if isinstance(s["outputs"], str) \
                        else " ".join(s["outputs"])
                    ems.append(100.0 * all(g.strip() in (pred or "")
                                           for g in (s["outputs"] if
                                                     isinstance(s["outputs"], list)
                                                     else [s["outputs"]])))
                em = sum(ems) / max(len(ems), 1)
                nl = sum(nlls) / max(len(nlls), 1)
                res[f"{task}@{n}"] = {"em": em, "nll_v": nl, "n": len(ems)}
                print(f"  {task}@{n}: EM {em:5.1f}%  NLL(values) {nl:.3f}  "
                      f"n={len(ems)}", flush=True)
        rows.append({"arm": tag, "alloc": alloc, "total_dc": total, "res": res,
                     "secs": round(time.perf_counter() - t0, 1)})
        json.dump(rows, open(a.out, "w"), indent=1)
        del m
        torch.cuda.empty_cache()
    print(f"\n  -> {a.out}", flush=True)
    print("  NOTE: no adapters on any arm, so these are pre-recovery numbers. "
          "They rank the allocations; they do not predict trained accuracy.",
          flush=True)


if __name__ == "__main__":
    main()
