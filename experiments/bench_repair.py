"""Re-generate ONLY the truncated items of a finished run, at a full budget.

WHY NOT JUST RE-RUN THE FAILURES. Re-running everything that failed and keeping
whatever now passes is best-of-2 applied to the losing subset. It can only move
the score up, it moves it up even for a model that got luckier rather than
better, and the resulting number is not a pass@1 on anything. It is the single
most tempting way to fabricate a benchmark improvement, so this script refuses
to do it: `--only truncated` is the only selection offered.

Truncation is different in kind. A generation that hit the cap was cut off by a
harness parameter, not by the model deciding it was finished -- it is a MISSING
measurement, not a failed one. Resuming it at a proper budget is a repair.

AND IT IS RESUMED, NOT RESAMPLED. The prompt plus the truncated text is
prefilled and decoding CONTINUES from where the cap stopped it. Autoregressive
sampling is sequential, so continuing from an identical prefix with an identical
sampler is exactly what an uninterrupted run would have produced -- this is
un-pausing the generation, not re-rolling it. Resampling from scratch would draw
a different trajectory that merely happens to be longer, which is a weaker claim
and needlessly discards the tokens already paid for. `--mode resample` exists
only to measure the difference between the two.

The discipline that keeps it a repair and not a fishing trip:
  * selection uses ONLY the token count, which is independent of correctness;
  * every re-generated item is re-scored and the new verdict STANDS, including
    when the item passed while truncated and fails on the retake;
  * both numbers are reported, with the count of items touched.

Truncated items are identified by re-tokenising the saved generation and
comparing with the original cap -- verified exact against the harness's own
counter (8/164 HumanEval, 72/257 MBPP).

    python experiments/bench_repair.py --gens logs/bench_nothink_full \
        --arms masked=ckpt/adapters-masked150-best.pt --tasks humaneval mbpp gsm8k \
        --old-max-new 768 --max-new 8192
"""
import argparse
import json
import os
import time

import torch

from experiments.bench_full import (GSM_PROMPT, HE_PROMPT, MBPP_PROMPT,
                                    extract_number, generate_batch, load_rows,
                                    score_humaneval, score_mbpp, strip_thinking)
from mercurius.paths import ROOT, STAGE_AB


def score(task, out, row):
    if task == "gsm8k":
        got = extract_number(strip_thinking(out))
        want = extract_number(row["answer"])
        return got is not None and want is not None and abs(got - want) < 1e-4
    return score_humaneval(out, row) if task == "humaneval" else score_mbpp(out, row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, help="prefix of the run to repair")
    ap.add_argument("--arms", nargs="+", required=True, metavar="NAME=CKPT")
    ap.add_argument("--tasks", nargs="+", default=["humaneval", "mbpp", "gsm8k"])
    ap.add_argument("--old-max-new", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--covs", default=str(ROOT / "cache/kv_covs_4b_mix.pt"))
    ap.add_argument("--mla-groups", default=str(ROOT / "cache/mla_groups_retr_4096_mix.json"))
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--only", default="truncated", choices=["truncated"],
                    help="the ONLY admissible selection; see the module docstring")
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except load the model and generate: "
                         "proves the selection, prompts and join are sound "
                         "before a long run commits the GPU")
    ap.add_argument("--mode", default="continue", choices=["continue", "resample"],
                    help="continue: prefill prompt+partial and resume decoding "
                         "(exact, keeps the sample). resample: start over.")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from mercurius import guard
    from mercurius.eval.retrieval_ab import build, build_original_nf4
    guard.cap_cuda_memory(60)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_ids = {tok.eos_token_id}
    for s in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0:
            eos_ids.add(i)

    out_report = {}
    for spec in a.arms:
        name, ckpt = spec.split("=", 1)
        model = None
        for task in a.tasks:
            gp = f"{a.gens}_{name}_{task}_nothink_gens.jsonl"
            if not os.path.exists(gp):
                print(f"  skip {name}/{task}: no generations at {gp}", flush=True)
                continue
            gens = {g["idx"]: g for g in map(json.loads, open(gp))}
            rows = load_rows(task, 0)
            # selection by BUDGET EXHAUSTION ONLY -- never by pass/fail.
            # Runs after 2026-09-24 record the flag; older ones are reconstructed
            # by re-tokenising, which was verified exact against the harness
            # counter (8/164 humaneval, 72/257 mbpp).
            if any("truncated" in g for g in gens.values()):
                trunc = sorted(i for i, g in gens.items() if g.get("truncated"))
            else:
                cut = a.old_max_new - 8
                trunc = sorted(i for i, g in gens.items()
                               if len(tok(g["out"], add_special_tokens=False).input_ids) >= cut)
            before = sum(score(task, g["out"], rows[i]) for i, g in sorted(gens.items()))
            n = len(gens)
            print(f"\n=== {name}/{task}: {before}/{n} = {before/n:.1%} before; "
                  f"{len(trunc)} truncated to re-take at {a.max_new}", flush=True)
            if not trunc:
                out_report.setdefault(name, {})[task] = {
                    "before": before, "after": before, "n": n, "repaired": 0}
                continue
            fmt = (lambda r: HE_PROMPT.format(prompt=r["prompt"])) if task == "humaneval" \
                else (lambda r: MBPP_PROMPT.format(text=r["prompt"], test=r["test_list"][0])) \
                if task == "mbpp" else (lambda r: GSM_PROMPT.format(question=r["question"]))
            prompts = {i: tok.apply_chat_template(
                [{"role": "user", "content": fmt(rows[i])}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for i in trunc}
            if a.mode == "continue":
                # prefill the partial answer so decoding RESUMES mid-sentence
                texts = {i: prompts[i] + gens[i]["out"] for i in trunc}
                # the tokens already spent do not come out of the new budget
                budget = a.max_new - a.old_max_new
            else:
                texts, budget = prompts, a.max_new
            # the join must not retokenise the prompt differently, or the
            # resumed context is not the context the model actually saw
            drift = sum(1 for i in trunc
                        if tok(texts[i], add_special_tokens=False).input_ids[
                            :len(tok(prompts[i], add_special_tokens=False).input_ids)]
                        != tok(prompts[i], add_special_tokens=False).input_ids)
            if drift:
                print(f"  NOTE: {drift}/{len(trunc)} prompts retokenise differently "
                      f"at the join; those resume from a near- not exact context",
                      flush=True)
            order = sorted(trunc, key=lambda i: len(texts[i]))
            if a.dry_run:
                # NOTE: some truncations are degenerate repetition loops rather
                # than answers that needed more room. Resuming those consumes the
                # whole new budget and still fails. That is a real property of
                # the model at these sampler settings, so it is reported
                # (`still_truncated`) rather than patched around with a
                # repetition penalty -- changing the sampler would break the
                # "resume is exactly an uninterrupted run" guarantee.
                ex = texts[order[0]]
                print(f"  DRY RUN: {len(trunc)} to resume, budget {budget} more "
                      f"tokens, longest context {max(len(tok(texts[i], add_special_tokens=False).input_ids) for i in trunc)} tok")
                print(f"  resume point (last 120 chars of the prefilled text):\n"
                      f"    ...{ex[-120:]!r}")
                continue
            if model is None:
                model = (build_original_nf4() if ckpt in ("original", "base", "none")
                         else build(ckpt, a.dc, a.covs, quantize=True, groups=a.mla_groups))
                model.eval()
            t0, still, flips = time.time(), 0, {"pass_to_fail": 0, "fail_to_pass": 0}
            for b0 in range(0, len(order), a.batch):
                idxs = order[b0:b0 + a.batch]
                outs, nt, _cut = generate_batch(
                    model, tok, [texts[i] for i in idxs], budget, eos_ids,
                    a.temperature, a.top_p, a.top_k)
                if a.mode == "continue":      # stitch the tail back on
                    outs = [gens[i]["out"] + o for i, o in zip(idxs, outs)]
                still += nt
                for i, o in zip(idxs, outs):
                    was = score(task, gens[i]["out"], rows[i])
                    now = score(task, o, rows[i])          # the retake STANDS
                    if was and not now:
                        flips["pass_to_fail"] += 1
                    elif now and not was:
                        flips["fail_to_pass"] += 1
                    gens[i]["out"] = o
                    gens[i]["repaired"] = a.mode
                print(f"  repaired {min(b0+a.batch,len(order))}/{len(order)}  "
                      f"still truncated {still}  {(time.time()-t0)/60:.1f} min", flush=True)
            after = sum(score(task, g["out"], rows[i]) for i, g in sorted(gens.items()))
            with open(gp.replace(".jsonl", "_repaired.jsonl"), "w") as fh:
                for i in sorted(gens):
                    fh.write(json.dumps(gens[i]) + "\n")
            print(f"  {name}/{task}: {before}/{n} = {before/n:.1%}  ->  "
                  f"{after}/{n} = {after/n:.1%}   ({flips['fail_to_pass']} fail->pass, "
                  f"{flips['pass_to_fail']} pass->fail, {still} still truncated)", flush=True)
            out_report.setdefault(name, {})[task] = {
                "before": before, "after": after, "n": n, "repaired": len(trunc),
                "still_truncated": still, "flips": flips, "max_new": a.max_new}
            json.dump(out_report, open(f"{a.gens}_repaired.json", "w"), indent=1)
        if model is not None:
            del model
            torch.cuda.empty_cache()
    print(f"\n-> {a.gens}_repaired.json")


if __name__ == "__main__":
    main()
