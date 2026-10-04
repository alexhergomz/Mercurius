"""HumanEval / MBPP / GSM8K on our arms -- comparable numbers, not just our own.

The held-out task split (experiments/capability_eval.py) measures the capability
our corpus targets, on tasks we own. These three measure something different and
equally necessary: whether the model is any good by numbers other people report.
A benchmark we built ourselves can always be accused of being easy in the ways
our data is strong.

Licences, for evaluation use: HumanEval MIT, MBPP CC-BY-4.0, GSM8K MIT. All three
are used ONLY for measurement, never for training, and are recorded as such in
data/ATTRIBUTION.md alongside WikiText and PG-19.

Contamination caveat, stated plainly: the base Qwen3.5-4B was pretrained on an
undisclosed corpus and these benchmarks are old and widely mirrored, so a good
score partly measures memorisation. That is why the held-out split matters more
for our purposes -- it is built from repositories at commits the base model may
have seen, but from symbols and questions that exist nowhere else. These numbers
are for comparability, not for truth.

Execution: HumanEval and MBPP need to RUN generated code. That happens in a
subprocess with a timeout and no network, which is enough for benchmark
solutions and is not a security boundary -- do not point this at untrusted
generations.

    python experiments/bench_standard.py --arms pilot=ckpt/adapters-pilotmix-best.pt \
        --tasks humaneval gsm8k --limit 40
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter

import torch

from mercurius.paths import LOGS_DIR, ROOT, STAGE_AB

HE_PROMPT = "Complete the following Python function.\n\n```python\n{prompt}```\n"
MBPP_PROMPT = ("Write a Python function for this task.\n\n{text}\n\n"
               "It must satisfy:\n```python\n{test}\n```\n")
GSM_PROMPT = ("Solve the problem. Reason briefly, then give the final numeric "
              "answer on its own last line as `#### <number>`.\n\n{question}\n")


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m[0] if m else text


def extract_number(text):
    """Last number in the text, preferring an explicit `#### n`.

    Returns a float or None. The regex could previously match a bare "-" or a
    trailing "." and float() then raised, which killed the whole run after
    HumanEval had already finished and before anything was written.
    """
    for pat in (r"####\s*(-?\d[\d,]*(?:\.\d+)?)", r"(-?\d[\d,]*(?:\.\d+)?)"):
        m = re.findall(pat, text or "")
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                continue
    return None


def run_python(code, timeout=12):
    """Subprocess with a timeout. Not a sandbox -- benchmark code only."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        p = fh.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True,
                           timeout=timeout, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(p)


def load_tasks(name, limit):
    from datasets import load_dataset
    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    else:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)[:limit] if limit else list(ds)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, metavar="NAME=CKPT")
    ap.add_argument("--tasks", nargs="+", default=["humaneval", "gsm8k"],
                    choices=["humaneval", "mbpp", "gsm8k"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=420)
    ap.add_argument("--covs", default=str(ROOT / "cache/kv_covs_4b_mix.pt"))
    ap.add_argument("--mla-groups", default=str(ROOT / "cache/mla_groups_retr_4096_mix.json"))
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--out", default=str(LOGS_DIR / "bench_standard.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from mercurius.eval.retrieval_ab import build
    from mercurius import guard
    from experiments.score_capability import gen_turn
    guard.cap_cuda_memory(60)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    report = {}
    for spec in a.arms:
        name, ckpt = spec.split("=", 1)
        print(f"\n=== {name}", flush=True)
        # "original" = the unmodified base model, no surgery and no adapters.
        # The only comparison that says what the surgery cost, since every other
        # arm shares the same conversion.
        if ckpt in ("original", "base", "none"):
            from mercurius.eval.retrieval_ab import build_original_nf4
            model = build_original_nf4().eval()
        else:
            model = build(ckpt, a.dc, a.covs, quantize=True, groups=a.mla_groups).eval()
        report[name] = {}
        for task in a.tasks:
            rows = load_tasks(task, a.limit)
            st, t0 = Counter(), time.time()
            for i, r in enumerate(rows, 1):
                if task == "humaneval":
                    msg = HE_PROMPT.format(prompt=r["prompt"])
                elif task == "mbpp":
                    msg = MBPP_PROMPT.format(text=r["prompt"], test=r["test_list"][0])
                else:
                    msg = GSM_PROMPT.format(question=r["question"])
                text = tok.apply_chat_template([{"role": "user", "content": msg}],
                                               tokenize=False, add_generation_prompt=True)
                try:
                    out = gen_turn(model, tok, text, max_new=a.max_new)
                except Exception:
                    st["gen_error"] += 1
                    continue
                if task == "gsm8k":
                    got, want = extract_number(out), extract_number(r["answer"])
                    ok = got is not None and want is not None and abs(got - want) < 1e-4
                elif task == "humaneval":
                    ok = run_python(extract_code(out) + "\n" + r["test"]
                                    + f"\ncheck({r['entry_point']})\n")
                else:
                    ok = run_python(extract_code(out) + "\n"
                                    + "\n".join(r["test_list"]) + "\n")
                st["pass" if ok else "fail"] += 1
                if i % 10 == 0:
                    print(f"  {task} {i}/{len(rows)}  pass {st['pass']}/{i} "
                          f"({st['pass'] / i:.0%})  {(time.time() - t0) / 60:.1f} min",
                          flush=True)
            n = st["pass"] + st["fail"]
            report[name][task] = {"pass": st["pass"], "n": n,
                                  "rate": st["pass"] / max(n, 1), **st}
            print(f"  {name} {task}: {st['pass']}/{n} = "
                  f"{st['pass'] / max(n, 1):.1%}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
            # write after EVERY task: HumanEval's 26 minutes were lost once to a
            # crash in the gsm8k scorer that ran afterwards
            json.dump(report, open(a.out, "w"), indent=1)
        del model
        torch.cuda.empty_cache()
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
