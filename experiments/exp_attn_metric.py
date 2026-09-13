"""Does an attention-aware decomposition objective beat CARE at d_c=256?

Four arms, no training -- this is a surgery-quality question, the same shape as
the CARE-vs-plain-SVD test that measured 26.76 pp.

  base      CARE as shipped: input whitening by C = E[xx^T], scalar K/V balance
  vmet      + block-diagonal OUTPUT metric M_V = sum_h W_O[:,h]^T W_O[:,h].
            Free -- built from o_proj's weights, no calibration. Two-sided
            weighted low-rank with a separable weight is still ONE SVD
            (Manton/Mahony/Hua 2003 Thm 3; Markovsky 2019 Thm 4.12).
  acov      + attention-weighted INPUT covariance. What reaches the residual
            stream is P V, so the exact metric for the V path is
            C_V = sum_h (P_h X)^T (P_h X), verified to machine precision.
            P is column-centred first -- without that, attention sinks collapse
            C_V to effective rank ~1 (see attn_covs.py).
  both      vmet + acov

Reference arms: the uncompressed model, and plain (unwhitened) SVD, so the
result is placed against the gap whitening already closed. From logs/care.json
that gap is 16.705 -> 13.398 against a 12.36 uncompressed baseline: whitening
took 76% of it, leaving ~1.04 ppl of headroom for everything measured here.
A refinement worth <1 ppl is the realistic ceiling, so read small numbers as
small, not as failure.
"""
import sys, json, time, argparse, torch
from transformers import AutoTokenizer
from mercurius.calibration.care import build, CKPT, CALIB
from mercurius.calibration.attention import collect_attn_covariances, mix
from mercurius.surgery.transmla import convert_to_mla
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.paths import WIKITEXT

LENGTHS = [2048, 8192]
GAPS = [4096, 16384]
DATA = str(WIKITEXT)


def evaluate(model, ids, needle, label):
    row = {"config": label, "ppl": {}, "retr": {}}
    for n in LENGTHS:
        row["ppl"][n] = perplexity(model, ids, n)
        torch.cuda.empty_cache()
    for g in GAPS:
        row["retr"][g] = retrieval(model, ids, needle, g)[2]
        torch.cuda.empty_cache()
    print(f"  {label:<10}" + "".join(f"{row['ppl'][n]:>10.3f}" for n in LENGTHS)
          + "".join(f"{row['retr'][g]:>10.3f}" for g in GAPS), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--acov", default="cache/attn_covs.pt")
    ap.add_argument("--out", default="logs/attn_metric.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    calib = tok(open(CALIB).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    # --- one calibration pass gives BOTH statistics on the SAME samples ---
    import os
    if os.path.exists(a.acov):
        print(f"loading {a.acov}", flush=True)
        covs = torch.load(a.acov)
    else:
        print(f"collecting covariances: {a.samples} x {a.seq} "
              f"(eager attention, both statistics in one pass)", flush=True)
        m = build()
        t0 = time.perf_counter()
        covs = collect_attn_covariances(m, calib, a.samples, a.seq)
        print(f"  took {time.perf_counter()-t0:.0f}s", flush=True)
        del m; torch.cuda.empty_cache()
        torch.save(covs, a.acov)

    # how different are the two statistics, before spending any GPU on evals?
    print("\n  layer   eff-rank(C_x)  eff-rank(C_v)   subspace overlap@256", flush=True)
    for i in sorted(covs):
        cx, cv = covs[i]["x"].double(), covs[i]["v"].double()
        def eff(c):
            w = torch.linalg.eigvalsh(c).clamp_min(0)
            p = w / w.sum().clamp_min(1e-30)
            return float(torch.exp(-(p * (p + 1e-30).log()).sum()))
        ex, ev = eff(cx), eff(cv)
        Ux = torch.linalg.eigh(cx).eigenvectors[:, -a.dc:]
        Uv = torch.linalg.eigh(cv).eigenvectors[:, -a.dc:]
        ov = float((Ux.T @ Uv).pow(2).sum() / a.dc)
        print(f"  {i:>5}  {ex:>13.1f}  {ev:>13.1f}   {ov:>18.3f}", flush=True)

    rows = []
    hdr = (f"\n  {'arm':<10}" + "".join(f"{'ppl@'+str(n):>10}" for n in LENGTHS)
           + "".join(f"{'retr@'+str(g//1024)+'k':>10}" for g in GAPS))
    print(hdr + "\n  " + "-" * (len(hdr) - 3), flush=True)

    C_x = mix(covs, "x")
    C_v = mix(covs, "v")

    m = build(); rows.append(evaluate(m, ids, needle, "uncompressed"))
    del m; torch.cuda.empty_cache()

    for label, cov, vmet in (("base", C_x, False),
                             ("vmet", C_x, True),
                             ("acov", C_v, False),
                             ("both", C_v, True)):
        m = build()
        convert_to_mla(m, d_c=a.dc, covs=cov, v_metric=vmet, verbose=False)
        rows.append(evaluate(m, ids, needle, label))
        del m; torch.cuda.empty_cache()

    base = next(r for r in rows if r["config"] == "base")
    print("\n  === vs the CARE baseline (negative = better) ===", flush=True)
    for r in rows:
        if r["config"] in ("base", "uncompressed"):
            continue
        d = "".join(f"{(r['ppl'][n]/base['ppl'][n]-1)*100:>+9.2f}%" for n in LENGTHS)
        dr = "".join(f"{(r['retr'][g]-base['retr'][g]):>+9.3f}" for g in GAPS)
        print(f"  {r['config']:<10}{d}{dr}", flush=True)

    json.dump({"dc": a.dc, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
