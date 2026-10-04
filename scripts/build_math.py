"""Build the long-run MATH data (#61): text chain-of-thought only, policy-clean.

  gsm8k-human   A. GSM8K TRAIN with its HUMAN-WRITTEN solutions (MIT). Calculator
                   annotations <<48/2=24>> are stripped; the native final "#### N" is
                   kept. No model anywhere.
  omi-text      C. nvidia/OpenMathInstruct-1 GSM8K-derived rows with is_correct AND no
                   <llm-code> (text CoT; 98% of rows are code-interpreter style and are
                   EXCLUDED -- #61), at most --per-question per question.
  teacher       B/D. problems -> solutions written by our Apache-2.0 teacher
                   (llama.cpp, /completion), CONDITIONED ON THE KNOWN ANSWER and kept
                   only if the final answer matches. --problems gsm8k-train (B) or a
                   JSONL of procedural problems (D, see procedural).
  procedural    D. problems from google-deepmind/mathematics_dataset (Apache-2.0,
                   fully procedural, no model) with their computed answers.
  pack          all rows -> multi-problem chat sequences of ~--target tokens, so the
                   episode sampler's minimum length holds; one output JSONL.

Every row carries `source`, `generator` and `license`. The user turn is the QUESTION
ONLY -- not our GSM8K eval instruction -- so training does not copy the benchmark prompt.
Decontaminate the packed output with experiments/decontam.py before use.

    .venv/bin/python scripts/build_math.py gsm8k-human --out data/math/gsm8k_human.jsonl
    .venv/bin/python scripts/build_math.py omi-text   --out data/math/omi_text.jsonl
    .venv/bin/python scripts/build_math.py pack data/math/*.jsonl --out data/math/math_packed.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.request

CALC = re.compile(r"<<[^>]*>>")


def write(rows, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):,} rows -> {out}", flush=True)


def row(q, a, source, generator, license_):
    return {"messages": [{"role": "user", "content": q.strip()},
                         {"role": "assistant", "content": a.strip()}],
            "source": source, "generator": generator, "license": license_}


def gsm8k_answer(sol):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", sol)
    return m.group(1).replace(",", "") if m else None


def cmd_gsm8k_human(a):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = [row(r["question"], CALC.sub("", r["answer"]), "openai/gsm8k:train",
                "human (GSM8K contractors)", "MIT") for r in ds]
    write(rows, a.out)


def cmd_omi_text(a):
    from datasets import load_dataset
    ds = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
    per, rows, seen, code = {}, [], 0, 0
    for r in ds:
        if r.get("dataset") != "gsm8k" or not r.get("is_correct"):
            continue
        seen += 1
        sol = r["generated_solution"]
        if "<llm-code" in sol:
            code += 1
            continue
        q = r["question"]
        if per.get(q, 0) >= a.per_question:
            continue
        per[q] = per.get(q, 0) + 1
        rows.append(row(q, sol, "nvidia/OpenMathInstruct-1:gsm8k:text-only",
                        "mistralai/Mixtral-8x7B-Instruct-v0.1", "NVIDIA License (permissive, "
                        "non-viral; data_policy 12.2)"))
        if seen % 100000 == 0:
            print(f"  scanned {seen:,} gsm8k-correct rows, code {code:,}, kept {len(rows):,}",
                  flush=True)
    print(f"  gsm8k-correct rows {seen:,}: code-interpreter {code:,} "
          f"({100*code/max(seen,1):.1f}%) excluded; kept {len(rows):,} text rows over "
          f"{len(per):,} questions", flush=True)
    write(rows, a.out)


def cmd_procedural(a):
    """mathematics_dataset: problems + computed answers. The package is imported at use
    so the rest of the script works without it."""
    try:
        # 2019 package vs modern sympy: base_solution_linear moved into
        # sympy.solvers.diophantine.diophantine. Re-export it where the package looks.
        # (`import sympy.solvers.diophantine` yields the FUNCTION in sympy 1.14, so the
        # subpackage must be patched through sys.modules.)
        import numpy as _np                     # aliases removed in NumPy >= 1.24
        for _n, _t in (("object", object), ("int", int), ("float", float), ("bool", bool)):
            if not hasattr(_np, _n):
                setattr(_np, _n, _t)
        import importlib as _il
        _m = _il.import_module("sympy.solvers.diophantine.diophantine")
        sys.modules["sympy.solvers.diophantine"].base_solution_linear = _m.base_solution_linear
        from mathematics_dataset import generate as mg
        from mathematics_dataset.modules import modules as mm
    except ImportError:
        sys.exit("pip install mathematics_dataset (Apache-2.0) first")
    rng = random.Random(a.seed)
    import numpy as np
    np.random.seed(a.seed)
    mods = mm.train(mg._make_entropy_fn(0, 1))            # difficulty 'train-medium'
    names = sorted(k for k in _flatten(mods) if not any(x in k for x in a.exclude))
    flat = _flatten(mods)
    out, bad, errs = [], set(), {}
    while len(out) < a.n and len(bad) < len(names):
        k = names[rng.randrange(len(names))]
        if k in bad:
            continue
        try:
            prob = flat[k]()
        except Exception as e:
            # some 2019 generators pass floats to random.randint, which Python >= 3.10
            # rejects; drop a module after 3 failures rather than patch its internals
            errs[k] = errs.get(k, 0) + 1
            if errs[k] >= 3:
                bad.add(k)
                print(f"  dropping module {k}: {type(e).__name__}: {e}", flush=True)
            continue
        out.append({"question": str(prob.question), "answer": str(prob.answer),
                    "module": k, "source": "google-deepmind/mathematics_dataset",
                    "license": "Apache-2.0"})
    print(f"  {len(out):,} problems from {len(names) - len(bad)} of {len(names)} modules",
          flush=True)
    write(out, a.out)


def _flatten(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(_flatten(v, f"{pre}{k}__"))
        else:
            o[f"{pre}{k}"] = v
    return o


def _norm_num(s):
    s = s.strip().rstrip(".").replace(",", "").replace("$", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


def cmd_teacher(a):
    """Teacher-written CoT, conditioned on the known answer, verified by answer match.
    --workers parallel requests: pair with a llama-server started with --parallel >= it."""
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    from mercurius.paths import STAGE_AB
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    if a.problems == "gsm8k-train":
        from datasets import load_dataset
        probs = [{"question": r["question"], "answer": gsm8k_answer(r["answer"]),
                  "source": "openai/gsm8k:train", "license": "MIT"}
                 for r in load_dataset("openai/gsm8k", "main", split="train")]
    else:
        probs = [json.loads(l) for l in open(a.problems, encoding="utf-8")]
    if a.limit:
        probs = probs[:a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["messages"][0]["content"] for l in open(a.out, encoding="utf-8")}
    todo = [p for p in probs if p["question"].strip() not in done]
    print(f"teacher: {len(todo):,} problems to do ({len(done):,} already done)", flush=True)

    def one(p):
        instr = (f"{p['question'].strip()}\n\n(The correct final answer is {p['answer']}. "
                 "Write a clear step-by-step solution that arrives at it, then give the "
                 "final answer on its own last line as `#### <answer>`. Do not mention "
                 "that the answer was provided.)")
        prompt = tok.apply_chat_template([{"role": "user", "content": instr}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        body = json.dumps({"prompt": prompt, "n_predict": a.max_new, "temperature": 0.3,
                           "top_p": 0.9, "cache_prompt": False}).encode()
        req = urllib.request.Request(a.server.rstrip("/") + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            sol = json.loads(urllib.request.urlopen(req, timeout=900).read())["content"]
        except Exception as e:
            return p, None, f"request failed: {e}"
        m = re.findall(r"####\s*(.+)", sol)
        if not m or _norm_num(m[-1]) != _norm_num(str(p["answer"])):
            return p, None, "answer mismatch"
        if re.search(r"(was|were) (given|provided)|the correct (final )?answer is", sol, re.I):
            return p, None, "leaks the conditioning"
        return p, sol, None

    kept = tried = fails = 0
    reasons = {}
    with open(a.out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        for p, sol, why in ex.map(one, todo):
            tried += 1
            if sol is None:
                reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1
                fails += why.startswith("request failed")
                if fails >= 20:
                    print("  20 request failures -- stopping; rerun to resume", flush=True)
                    break
                continue
            r = row(p["question"], sol, p["source"] + ":teacher",
                    "Qwen3.5-35B-A3B (Apache-2.0), answer-conditioned", p["license"])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
            kept += 1
            if tried % 500 == 0:
                print(f"  {tried:,} tried, {kept:,} kept ({100*kept/tried:.0f}%)  "
                      f"rejects {reasons}", flush=True)
    print(f"teacher: {tried:,} tried, {kept:,} kept -> {a.out}  rejects {reasons}", flush=True)


def cmd_pack(a):
    """Pack single-problem rows into multi-problem conversations of ~target tokens."""
    from transformers import AutoTokenizer
    from mercurius.paths import STAGE_AB
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    rows = []
    for p in a.inputs:
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    random.Random(a.seed).shuffle(rows)
    packs, cur, cur_len, srcs = [], [], 0, set()
    for r in rows:
        # tokenize=True returns a dict (BatchEncoding) in current transformers, whose
        # len() is the number of KEYS -- render the text and tokenize explicitly
        n = len(tok(tok.apply_chat_template(r["messages"], tokenize=False),
                    add_special_tokens=False).input_ids)
        if cur and cur_len + n > a.target:
            packs.append(_pack(cur, srcs)); cur, cur_len, srcs = [], 0, set()
        cur.append(r); cur_len += n; srcs.add((r["source"], r["generator"], r["license"]))
    if cur:
        packs.append(_pack(cur, srcs))
    write(packs, a.out)


def _pack(rows, srcs):
    msgs = [m for r in rows for m in r["messages"]]
    return {"messages": msgs, "n_problems": len(rows),
            "provenance": [{"source": s, "generator": g, "license": l}
                           for s, g, l in sorted(srcs)],
            "license": "; ".join(sorted({l for _, _, l in srcs}))}


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gsm8k-human"); p.add_argument("--out", required=True)
    p = sp.add_parser("omi-text"); p.add_argument("--out", required=True)
    p.add_argument("--per-question", type=int, default=2)
    p = sp.add_parser("procedural"); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", nargs="*", default=[])
    p = sp.add_parser("teacher"); p.add_argument("--out", required=True)
    p.add_argument("--problems", required=True, help="gsm8k-train or a problems JSONL")
    p.add_argument("--server", default="http://127.0.0.1:8077")
    p.add_argument("--max-new", type=int, default=768); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("pack"); p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True); p.add_argument("--target", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"gsm8k-human": cmd_gsm8k_human, "omi-text": cmd_omi_text, "procedural": cmd_procedural,
     "teacher": cmd_teacher, "pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
