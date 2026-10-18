"""Validate every benchmark scorer before trusting a single number from it.

Eight measurement artifacts in one day -- truncated generations, a JSON-vs-XML
parse mismatch, missing test dependencies, PEP 735 groups, a crash that discarded
finished work, `rows[:25]` treated as a sample, a scorer that failed half of
HumanEval on missing imports -- every one of which produced a plausible, wrong,
actionable number. So the harness is now tested the way code is tested.

Two directions, both necessary:

  SENSITIVITY  a KNOWN-CORRECT answer must score as a pass. A scorer that fails
               correct answers understates the model, which is how 49.4% appeared
               where ~70% was expected.
  SPECIFICITY  a KNOWN-WRONG answer must score as a fail. A scorer that passes
               everything is worse than no scorer, because it looks like success.

Correct answers are perturbed into the shapes a real model actually emits: inside
a fence, with prose around it, without the prompt's imports, with the whole prompt
echoed back. Each of those has broken something already.

    python experiments/validate_benchmarks.py
"""
import re
import sys

sys.path.insert(0, ".")

from experiments.bench_full import (extract_code, extract_number, run_python,
                                    score_humaneval, score_mbpp)

FENCE = "```python\n{}\n```"
# The shapes that broke things AFTER the suite first passed. The suite missed
# the multi-fence case because it only ever built inputs with ONE fence, so it
# could not see that `extract_code` returns m[0]. 44 of 257 MBPP generations had
# more than one fence and one had 74. A validator only catches the shapes it
# thinks to build, so every shape found in real generations gets added here.
DRAFT_THEN_FINAL = ("Let me try.\n\n```python\ndef f():\n    return None  # wrong\n"
                    "```\n\nThat is wrong, let me redo it.\n\n```python\n{}\n```")
TRUNCATED = "Here is the function.\n\n```python\n{}\n\n# and now I keep going"
PROSE = ("Here is my solution. I first considered the edge cases, then wrote "
         "the function.\n\n```python\n{}\n```\n\nThis handles the empty case too.")


def strip_imports(text):
    return re.sub(r"^(?:from|import)\s+.*$\n?", "", text, flags=re.M)


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} got {got}, want {want}")
    return ok


def validate_humaneval(n=30):
    from datasets import load_dataset
    ds = list(load_dataset("openai/openai_humaneval", split="test"))[:n]
    print(f"\n=== HumanEval scorer ({n} problems)")
    res = []
    # sensitivity: the canonical solution, in the shapes a model emits
    for label, fn in (
        ("canonical, prompt+solution in fence",
         lambda r: FENCE.format(r["prompt"] + r["canonical_solution"])),
        ("canonical, imports STRIPPED",
         lambda r: FENCE.format(strip_imports(r["prompt"]) + r["canonical_solution"])),
        ("canonical, wrapped in prose",
         lambda r: PROSE.format(r["prompt"] + r["canonical_solution"])),
        ("canonical, NO fence at all",
         lambda r: r["prompt"] + r["canonical_solution"]),
    ):
        k = sum(score_humaneval(fn(r), r) for r in ds)
        res.append(check(f"sensitivity: {label}", k, n))
    k = sum(score_humaneval(DRAFT_THEN_FINAL.format(
        r["prompt"] + r["canonical_solution"]), r) for r in ds)
    res.append(check("sensitivity: DRAFT fence then correct fence", k, n))
    k = sum(score_humaneval(TRUNCATED.format(
        r["prompt"] + r["canonical_solution"]), r) for r in ds)
    res.append(check("sensitivity: correct code, fence never closed", k, n))
    # specificity: a function that returns the wrong thing must fail
    wrong = sum(score_humaneval(
        FENCE.format(f"def {r['entry_point']}(*args, **kwargs):\n    return None"), r)
        for r in ds)
    res.append(check("specificity: stub returning None", wrong, 0))
    empty = sum(score_humaneval("I am not sure how to do this.", r) for r in ds)
    res.append(check("specificity: no code at all", empty, 0))
    return all(res)


def validate_mbpp(n=30):
    from datasets import load_dataset
    ds = list(load_dataset("google-research-datasets/mbpp", "sanitized",
                           split="test"))[:n]
    print(f"\n=== MBPP scorer ({n} problems)")
    res = []
    k = sum(score_mbpp(FENCE.format(r["code"]), r) for r in ds)
    res.append(check("sensitivity: reference code in fence", k, n))
    k = sum(score_mbpp(PROSE.format(r["code"]), r) for r in ds)
    res.append(check("sensitivity: reference code in prose", k, n))
    k = sum(score_mbpp(DRAFT_THEN_FINAL.format(r["code"]), r) for r in ds)
    res.append(check("sensitivity: DRAFT fence then correct fence", k, n))
    k = sum(score_mbpp(TRUNCATED.format(r["code"]), r) for r in ds)
    res.append(check("sensitivity: correct code, fence never closed", k, n))
    wrong = sum(score_mbpp(DRAFT_THEN_FINAL.format("def zzz():\n    return None"), r)
                for r in ds)
    res.append(check("specificity: two fences, BOTH wrong", wrong, 0))
    wrong = sum(score_mbpp(FENCE.format("def f():\n    return None"), r) for r in ds)
    res.append(check("specificity: unrelated function", wrong, 0))
    return all(res)


def validate_gsm8k(n=300):
    from datasets import load_dataset
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))[:n]
    print(f"\n=== GSM8K extraction ({n} problems)")
    res = []
    # every gold answer must parse
    bad = [r for r in ds if extract_number(r["answer"]) is None]
    res.append(check("sensitivity: every gold answer parses", len(bad), 0))
    # a model echoing the gold in our requested format must score correct
    ok = 0
    for r in ds:
        want = extract_number(r["answer"])
        out = f"Let me work through it.\n\nSo the total is {want:g}.\n#### {want:g}"
        ok += extract_number(out) == want
    res.append(check("sensitivity: '#### n' after reasoning", ok, n))
    # common shapes a model actually produces
    cases = [("#### 18", 18.0), ("#### 1,234", 1234.0), ("The answer is 42", 42.0),
             ("#### -7", -7.0), ("#### 3.5", 3.5), ("$18 total\n#### 18", 18.0),
             ("no number here", None), ("", None), ("#### 18.", 18.0)]
    shape_ok = all(extract_number(t) == v for t, v in cases)
    res.append(check("shapes: ####, commas, negative, decimal, prose, empty",
                     shape_ok, True))
    # specificity: a wrong number must not match
    mism = 0
    for r in ds[:50]:
        want = extract_number(r["answer"])
        got = extract_number(f"#### {want + 1:g}")
        mism += (got is not None and abs(got - want) < 1e-4)
    res.append(check("specificity: off-by-one does NOT match", mism, 0))
    # the fallback picks the LAST number -- flag the known risk
    trail = extract_number("#### 18\n\nLet me double check: 18 is correct, unlike 25.")
    print(f"  NOTE  trailing-prose case -> {trail} "
          f"({'ok, #### wins' if trail == 18.0 else 'RISK: last-number fallback won'})")
    return all(res)


def main():
    results = {"humaneval": validate_humaneval(),
               "mbpp": validate_mbpp(),
               "gsm8k": validate_gsm8k()}
    print("\n=== summary")
    for k, v in results.items():
        print(f"  {k:<12} {'VALID' if v else 'BROKEN'}")
    if not all(results.values()):
        print("\nDo not benchmark until these pass.")
        sys.exit(1)
    print("\nAll scorers sensitive to correct answers and specific against wrong ones.")


if __name__ == "__main__":
    main()
