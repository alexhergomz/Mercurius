"""What did weight averaging actually buy? Raw step-600 weights vs the EMA.

Recoverable without a third training arm because the two artifacts differ ONLY in
the averaging:

  ckpt/adapters-<tag>.pt   final save, wrapped in ema.applied()  -> AVERAGED
  ckpt/resume-<tag>.pt     per-eval save, in ema.suspended()     -> RAW iterate

Both are written at step 600 by the same run, so the difference is the average and
nothing else. Needed because --ema-start defaults to steps//2, which makes the
in-training eval change character mid-run: the step-300 eval had 1 EMA point
(identical to the raw weights) and step-400 had 101, so the 9.420 -> 9.354 move
conflated "100 more steps" with "averaging switched on".

    python experiments/ema_contribution.py --tag recipe-moe600
"""
import argparse, sys, torch
sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--covs", default="cache/kv_covs_4b_mix.pt")
    ap.add_argument("--groups", default="cache/mla_groups_retr_4096_mix.json")
    ap.add_argument("--f2a2", action="store_true",
                    help="pass for an arm trained with --f2a2; build() detects it "
                         "from the artifact anyway, this is only a cross-check")
    a = ap.parse_args()
    from mercurius.eval.retrieval_ab import build
    from mercurius.eval.suite import ce_and_topk, report
    from mercurius.recovery.train import EVAL_DATA
    from mercurius.paths import STAGE_AB
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    ids = torch.tensor(tok(open(EVAL_DATA).read(),
                           add_special_tokens=False).input_ids)
    print(f"eval tokens: {len(ids):,}", flush=True)

    rows = {}
    for label, path in (("EMA average", f"ckpt/adapters-{a.tag}.pt"),
                        ("raw iterate", f"ckpt/resume-{a.tag}.pt")):
        import os
        if not os.path.exists(path):
            print(f"  SKIP {label}: {path} missing"); continue
        sd = torch.load(path, map_location="cpu")
        # the resume file nests the weights under "weights"; write a flat temp
        if "weights" in sd and isinstance(sd["weights"], dict):
            flat = sd["weights"]
            tmp = f"/tmp/claude-1000/_flat_{a.tag}.pt"
            torch.save(flat, tmp); path = tmp
            print(f"  ({label}: unwrapped resume payload, step {sd.get('step')})",
                  flush=True)
        m = build(path, a.dc, a.covs, quantize=True, groups=a.groups).eval()
        print(f"\n=== {label}  ({path})", flush=True)
        rows[label] = {}
        for n in (2048, 8192):
            r = ce_and_topk(m, ids, n)
            rows[label][n] = r
            report(f"@{n}", r)
        del m
        torch.cuda.empty_cache()

    if len(rows) == 2:
        print("\n=== EMA CONTRIBUTION (negative ppl delta = averaging helped)")
        for n in (2048, 8192):
            d = rows["EMA average"][n]["ppl"] - rows["raw iterate"][n]["ppl"]
            print(f"  @{n}: {rows['raw iterate'][n]['ppl']:.3f} raw -> "
                  f"{rows['EMA average'][n]['ppl']:.3f} averaged   "
                  f"delta {d:+.3f}")


if __name__ == "__main__":
    main()
