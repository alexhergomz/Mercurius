"""Compute the per-layer input covariances CARE whitening needs, and SAVE them.

run_care.py builds these inline and throws them away when it exits. Every MLA
run wants the same matrices, and recomputing means another calibration pass over
the GPU -- which on a single Orin is a pass we are not sharing with a training
job. The covariance is 6 layers x 1024 x 1024 float32 = 25 MB on disk, so the
right move is to compute once and load thereafter.

Reuses run_care.collect_covariances verbatim rather than reimplementing it, so
the matrices a training run whitens with are bit-identical to the ones the
CARE comparison was measured with. A second implementation here would be a
place for the two to silently drift apart.
"""
import os, shutil, argparse, torch
from transformers import AutoTokenizer
from mercurius.calibration.care import build, collect_covariances, CKPT, CALIB
from mercurius.paths import CACHE_DIR

MIN_FREE_GB = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=256,
                    help="CARE reports saturation past ~512; 256 is their default")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default=str(CACHE_DIR / 'kv_covs.pt'))
    a = ap.parse_args()

    # This machine's root fs is the only fs. Filling it bricks the Jetson, so
    # refuse rather than discover it at write time.
    free_gb = shutil.disk_usage("/").free / 1e9
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"refusing to write: only {free_gb:.1f} GB free "
                         f"(need {MIN_FREE_GB} GB headroom)")

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    calib = tok(open(CALIB).read(), return_tensors="pt").input_ids[0]
    print(f"calibration corpus {len(calib):,} tokens (fineweb-edu)", flush=True)

    m = build()
    print(f"collecting covariances: {a.samples} x {a.seq} tokens", flush=True)
    covs = collect_covariances(m, calib, a.samples, a.seq)
    del m
    torch.cuda.empty_cache()

    out = {int(k): v.cpu() for k, v in covs.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(out, a.out)
    mb = os.path.getsize(a.out) / 1e6
    print(f"\nwrote {a.out}  ({mb:.1f} MB, {len(out)} layers: "
          f"{sorted(out)})", flush=True)

    # sanity: each covariance must be symmetric PSD or the Cholesky in
    # whiten_factor falls back to the eigh path and quietly costs accuracy
    for i, c in sorted(out.items()):
        c = c.double()
        asym = (c - c.T).abs().max().item()
        w = torch.linalg.eigvalsh(c)
        print(f"  layer {i:>2}  asym {asym:.2e}  eig min {w.min():.3e} "
              f"max {w.max():.3e}  cond {(w.max()/w.min().clamp_min(1e-30)):.2e}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
