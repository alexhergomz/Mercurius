"""Decontaminate TRAINING sources against every EVAL set, by word 13-gram overlap.

WHY (user, 2026-09-29, long-run prep): before the long run, nothing we evaluate on may
leak into what we train or calibrate on. Math data makes this acute: the accepted
OpenMathInstruct-1 rows are built on GSM8K TRAIN problems while we score GSM8K TEST,
and self-generated problems must be checked the same way.

METHOD (the GPT-3 / Llama convention): lowercase, keep alphanumeric word tokens, and
flag a training document if it shares ANY 13-gram with an eval item. 13 words is long
enough that a match is not boilerplate, short enough to catch paraphrased copies.
Eval text uses the SAME loaders as the benchmarks (bench_full.load_rows, bench_mc), so
what is checked is exactly what is scored.

  eval sets    GSM8K test (question + answer), HumanEval (prompt + solution), MBPP
               sanitized test, WikiText eval file, the MC suite (6 tasks)
  sources      text corpora (documents split on blank lines, as training does) and
               episode / generic JSONL (every string field of a row)

Reports flagged documents per source x eval set, and with --write-clean writes a copy
of each source without the flagged documents (never modifies the original).

    .venv/bin/python experiments/decontam.py --text data/fineweb_edu_long.txt \
        --jsonl data/episodes/pilot_mix.jsonl --text data/synth_recall_4b.txt
"""
import argparse
import json
import os
import re
import sys

N = 13
WORD = re.compile(r"[a-z0-9]+")


def grams(text, n=N):
    w = WORD.findall(text.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def eval_texts():
    sys.path.insert(0, os.path.dirname(__file__))
    from bench_full import load_rows
    out = {}
    out["gsm8k"] = [r["question"] + " " + r["answer"] for r in load_rows("gsm8k", 0)]
    out["humaneval"] = [r["prompt"] + " " + r["canonical_solution"]
                        for r in load_rows("humaneval", 0)]
    out["mbpp"] = [r["prompt"] + " " + r["code"] for r in load_rows("mbpp", 0)]
    from mercurius.paths import WIKITEXT
    wt = open(WIKITEXT, encoding="utf-8", errors="replace").read()
    out["wikitext"] = [p for p in wt.split("\n\n") if p.strip()]
    try:
        from bench_mc import TASKS
        from datasets import load_dataset
        mc = []
        for name, (hf, cfg, split, build) in TASKS.items():
            ds = load_dataset(hf, cfg, split=split) if cfg else load_dataset(hf, split=split)
            for r in ds:
                ctx, conts, gold = build(r)
                ctx = ctx[0] if isinstance(ctx, list) else ctx
                mc.append(ctx + " " + conts[gold])
        out["mc_suite"] = mc
    except Exception as e:                       # the MC sets are optional here
        print(f"  (MC suite skipped: {e})", flush=True)
    return out


def text_docs(path):
    return [d for d in open(path, encoding="utf-8", errors="replace").read().split("\n\n")
            if d.strip()]


def jsonl_docs(path):
    docs = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        parts = []

        def walk(x):
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(row)
        docs.append((" ".join(parts), line))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", default=[], help="plain-text corpus")
    ap.add_argument("--jsonl", action="append", default=[], help="JSONL source")
    ap.add_argument("--write-clean", action="store_true",
                    help="write <source>.decon<ext> without flagged documents")
    ap.add_argument("--out", default="logs/decontam.json")
    a = ap.parse_args()

    ev = eval_texts()
    index = {}                                   # gram -> set of eval set names
    for name, items in ev.items():
        for t in items:
            for g in grams(t):
                index.setdefault(g, set()).add(name)
    print(f"eval 13-grams: {len(index):,} from " +
          ", ".join(f"{k} {len(v):,}" for k, v in ev.items()), flush=True)

    report = {}
    for kind, paths in (("text", a.text), ("jsonl", a.jsonl)):
        for p in paths:
            docs = text_docs(p) if kind == "text" else jsonl_docs(p)
            hits, flagged = {}, set()
            for i, d in enumerate(docs):
                body = d if kind == "text" else d[0]
                sets = set()
                for g in grams(body):
                    s = index.get(g)
                    if s:
                        sets |= s
                for s in sets:
                    hits[s] = hits.get(s, 0) + 1
                if sets:
                    flagged.add(i)
            report[p] = {"docs": len(docs), "flagged": len(flagged), "by_eval": hits}
            print(f"  {p}: {len(flagged):,} of {len(docs):,} documents flagged  "
                  f"{hits if hits else '(clean)'}", flush=True)
            if a.write_clean and flagged:
                root, ext = os.path.splitext(p)
                outp = f"{root}.decon{ext}"
                with open(outp, "w", encoding="utf-8") as fh:
                    if kind == "text":
                        fh.write("\n\n".join(d for i, d in enumerate(docs) if i not in flagged))
                    else:
                        fh.writelines(d[1] for i, d in enumerate(docs) if i not in flagged)
                print(f"    wrote {outp}", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
