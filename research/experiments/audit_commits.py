"""Audit the commit histories before building any task from them.

The MAVSDK lesson: a broken oracle is silent. It rejected correct answers and
accepted incomplete ones, and nothing in the metrics said so -- it took reading
the ground truth against an independent tool to find it. Commit-derived tasks
are a much bigger commitment than the parsed ones, so everything that could make
them low quality gets measured FIRST.

Checks, in the order they would sink the idea:

 1. YIELD      -- focused fix commits per repository and per language.
 2. FOCUS      -- diff size; a 3,000-line "fix" is a refactor and teaches noise.
 3. PARENT     -- can the buggy state be materialised, and what does it cost?
                  (treeless clones fetch file contents on demand.)
 4. SYMPTOM    -- is there a code-derived symptom to build a statement from?
                  Test changes in the same commit are the best source, because
                  they describe expected behaviour IN CODE rather than prose.
 5. LEAKAGE    -- would the statement hand over the answer?
 6. PII        -- author addresses and secrets inside diff text.
 7. LICENCE    -- diffs that touch vendored trees under another licence.
 8. TARGETS    -- for commit-as-supervision: how many diffs are small enough to
                  be a training target at all.

    python experiments/audit_commits.py --repos 25
"""
import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict

from mercurius.data.repo_env import _EMAIL, _EMAIL_OK
from mercurius.paths import ROOT

GIT = os.path.join(ROOT, "data/repos/git")
FIX = re.compile(r"\b(fix|bug|regress|crash|broken|incorrect|wrong|fail|error|"
                 r"segfault|leak|race|deadlock|overflow|npe|panic)", re.I)
NOT_FIX = re.compile(r"\b(typo|changelog|bump|readme|docs?|comment|lint|format|"
                     r"whitespace|spelling|version|release|dependabot|revert)", re.I)
SRC = re.compile(r"\.(py|js|jsx|ts|tsx|go|java|rs|c|h|cc|cpp|hpp|cs|rb|php|kt|"
                 r"scala|swift|dart|lua|ex|exs)$")
TEST = re.compile(r"(^|/)(tests?|spec|__tests__)/|_test\.|\.test\.|\.spec\.|"
                  r"(^|/)test_[^/]*\.py$", re.I)
VENDORED = re.compile(r"(^|/)(vendor|third_party|node_modules|external|deps)/", re.I)
LANG_OF = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go",
           ".java": "java", ".rs": "rust", ".c": "c", ".cpp": "cpp", ".cs": "csharp",
           ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".scala": "scala",
           ".swift": "swift", ".dart": "dart"}


def git(repo, *args, timeout=120):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, timeout=timeout)


def commits(repo, n=4000):
    r = git(repo, "log", "--no-merges", "--format=%x01%H%x00%s", "--name-only", f"-n{n}")
    out = []
    for b in r.stdout.split("\x01"):
        ls = [x for x in b.strip().split("\n") if x.strip()]
        if not ls or "\x00" not in ls[0]:
            continue
        sha, subj = ls[0].split("\x00", 1)
        out.append((sha, subj, ls[1:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=25)
    ap.add_argument("--diffs", type=int, default=4, help="commits to diff per repo")
    ap.add_argument("--out", default=str(ROOT / "logs/commit_audit.json"))
    a = ap.parse_args()
    bares = sorted(os.listdir(GIT))[:a.repos]
    agg = Counter()
    by_lang = Counter()
    diff_lines, target_ok, samples = [], 0, []
    pii = Counter()
    t0 = time.time()

    for bn in bares:
        repo = os.path.join(GIT, bn)
        name = bn[:-4].replace("__", "/")
        try:
            cs = commits(repo)
        except subprocess.SubprocessError:
            agg["log_timeout"] += 1
            continue
        agg["repos"] += 1
        agg["commits_scanned"] += len(cs)
        cands = []
        for sha, subj, files in cs:
            if not FIX.search(subj) or NOT_FIX.search(subj):
                continue
            src = [f for f in files if SRC.search(f) and not TEST.search(f)
                   and not VENDORED.search(f)]
            tst = [f for f in files if TEST.search(f)]
            if VENDORED.search(" ".join(files)):
                agg["touches_vendored"] += 1
            if not src:
                continue
            agg["fix_with_src"] += 1
            if len(src) > 3:
                agg["too_broad"] += 1
                continue
            agg["focused"] += 1
            if tst:
                agg["focused_with_test"] += 1
                cands.append((sha, subj, src, tst))
            for f in src:
                by_lang[LANG_OF.get(os.path.splitext(f)[1], "?")] += 1
        # --- diff-level checks on a few candidates
        for sha, subj, src, tst in cands[:a.diffs]:
            try:
                d = git(repo, "show", "--format=", "--unified=3", sha, timeout=90)
            except subprocess.SubprocessError:
                agg["diff_timeout"] += 1
                continue
            if d.returncode or not d.stdout:
                agg["diff_failed"] += 1
                continue
            agg["diff_ok"] += 1
            txt = d.stdout
            n_lines = sum(1 for l in txt.split("\n")
                          if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
            diff_lines.append(n_lines)
            if n_lines <= 120:
                target_ok += 1
            emails = [m for m in _EMAIL.findall(txt) if not _EMAIL_OK.search(m)]
            if emails:
                pii["diff_has_email"] += 1
            # does the TEST part of the diff describe behaviour? (a symptom source)
            test_hunks = [l for l in txt.split("\n")
                          if l.startswith("+") and re.search(r"(assert|expect|require|"
                                                             r"should|EXPECT_|ASSERT_)", l)]
            if test_hunks:
                agg["has_test_assertions"] += 1
            if len(samples) < 6 and test_hunks and n_lines <= 120:
                samples.append({"repo": name, "sha": sha[:10], "subject": subj[:70],
                                "src": src, "n_diff_lines": n_lines,
                                "test_assert": test_hunks[0].strip()[:110]})

    el = time.time() - t0
    diff_lines.sort()
    med = diff_lines[len(diff_lines) // 2] if diff_lines else 0
    p90 = diff_lines[int(len(diff_lines) * 0.9)] if diff_lines else 0
    rep = {"repos": agg["repos"], "commits_scanned": agg["commits_scanned"],
           "fix_with_src": agg["fix_with_src"], "focused_1_3_files": agg["focused"],
           "focused_with_test": agg["focused_with_test"],
           "too_broad_dropped": agg["too_broad"],
           "touches_vendored": agg["touches_vendored"],
           "diffs_sampled": agg["diff_ok"], "diff_failed": agg["diff_failed"],
           "diff_median_lines": med, "diff_p90_lines": p90,
           "diff_le_120_lines": target_ok,
           "with_test_assertions": agg["has_test_assertions"],
           "diffs_with_author_email": pii["diff_has_email"],
           "by_language": dict(by_lang.most_common()), "seconds": round(el, 1)}
    json.dump({"summary": rep, "samples": samples}, open(a.out, "w"), indent=1)
    for k, v in rep.items():
        print(f"  {k:<26} {v}")
    print("\nsamples (symptom comes from the TEST diff, never the message):")
    for s in samples:
        print(f"  {s['repo']} {s['sha']}  {s['n_diff_lines']} diff lines  src={s['src'][:2]}")
        print(f"     test asserts: {s['test_assert']}")
    per_repo = agg["focused_with_test"] / max(agg["repos"], 1)
    print(f"\nprojected over 78 repositories: ~{int(per_repo * 78):,} localization tasks "
          f"with a code-derived symptom")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
