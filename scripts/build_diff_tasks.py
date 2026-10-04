"""Commit diffs as training sequences: the cheapest real signal we have.

Every other tier costs teacher generation. This one costs nothing -- the target
is the fix a human already wrote, and the person who wrote it did the verifying.
~4,400 focused fix commits across the admitted repositories.

WHAT A SEQUENCE LOOKS LIKE

  user      the buggy source region, plus the test the commit added (which
            states the expected behaviour IN CODE -- that is the symptom
            statement, and it is why we require fix commits to touch tests)
  assistant the source half of the real diff

The commit MESSAGE is never used, in either half. It is human-authored prose,
and our line is that statements derive from code (docs/data_policy.md). Messages
serve only as the filter that finds fix commits, and they routinely name the file
anyway, which would make the task a copy exercise.

HOW THE TRAINER SHOULD READ THESE (decisions D10)

These are HUMAN targets, not teacher outputs. The excess-CE data term takes a
human target perfectly well -- the teacher's contribution there is a
gradient-free offset -- but the DIVERGENCE term would be asking the student to
match the teacher's opinion of human code, which may be poor and is not what we
want to teach. Records carry `"human_target": true` so the mix can weight the
two terms differently rather than treating a commit like a teacher rollout.

    python scripts/build_diff_tasks.py --repos 0 --per-repo 40
"""
import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter

from mercurius.data.exec_env import bare_of, split_diff, test_files_in
from mercurius.data.repo_env import scrub
from mercurius.paths import DATA_DIR, ROOT
from scripts.repo_licenses import PERMISSIVE

FIX = re.compile(r"\b(fix|bug|regress|crash|broken|incorrect|wrong|fail|error|"
                 r"segfault|leak|race|deadlock|overflow|npe|panic)", re.I)
NOT_FIX = re.compile(r"\b(typo|changelog|bump|readme|docs?|comment|lint|format|"
                     r"whitespace|spelling|version|release|dependabot|revert|merge)", re.I)
SRC = re.compile(r"\.(py|js|jsx|ts|tsx|go|java|rs|c|h|cc|cpp|hpp|cs|rb|php|kt|"
                 r"scala|swift|dart|lua|ex|exs)$")
TEST = re.compile(r"(^|/)(tests?|spec|__tests__)/|_test\.|\.test\.|\.spec\.|"
                  r"(^|/)test_[^/]*\.py$", re.I)
VENDORED = re.compile(r"(^|/)(vendor|third_party|node_modules|external|deps)/", re.I)

PROMPT = """The following test was added to `{test_path}` and it fails against the \
current code:

```
{test_body}
```

Here is the relevant code from `{src_path}`:

```
{src_before}
```

Fix the code so the test passes. Reply with a unified diff."""


def git(bare, *args, timeout=120):
    """Git output is not guaranteed UTF-8 -- a single stray byte in one repo's
    history killed the first run after 22 of 102 repositories."""
    r = subprocess.run(["git", "-C", bare, *args], capture_output=True, timeout=timeout)
    return r.returncode, r.stdout.decode("utf-8", "replace")


# a test FUNCTION, not the file's import block: the first run produced prompts
# that were nothing but imports, because the top of a test diff is where the
# imports live
_TESTSTART = re.compile(r"^\s*(?:@\w+[^\n]*\n\s*)*(?:async\s+)?(?:def|it|test|fn|"
                        r"func|public\s+void|@Test)\b", re.I)


def added_test_body(test_diff, max_lines=60):
    """The added block that looks like a test, preferring the first hunk that
    actually starts a test rather than the first hunk in the file."""
    blocks, cur = [], []
    for line in test_diff.split("\n"):
        if line.startswith("@@") or line.startswith("diff --git"):
            if cur:
                blocks.append("\n".join(cur))
            cur = []
            continue
        if line.startswith(("+++", "---", "index ")):
            continue
        if line.startswith("+"):
            cur.append(line[1:])
        elif line.startswith(" ") and cur:
            cur.append(line[1:])
    if cur:
        blocks.append("\n".join(cur))
    scored = [b for b in blocks if _TESTSTART.search(b) and len(b.strip()) > 60]
    pick = (scored or [b for b in blocks if len(b.strip()) > 60] or blocks or [""])[0]
    return "\n".join(pick.split("\n")[:max_lines])


def hunk_bodies(diff_text, max_lines=120):
    out, cur, path = {}, [], None
    for line in diff_text.split("\n"):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
        if m:
            if path and cur:
                out[path] = "\n".join(cur[:max_lines])
            path, cur = m.group(2), []
            continue
        if line.startswith(("+++", "---", "index ", "@@")):
            continue
        if line.startswith("+"):
            cur.append(line[1:])
        elif line.startswith(" ") and cur:
            cur.append(line[1:])
    if path and cur:
        out[path] = "\n".join(cur[:max_lines])
    return out


# files that are build plumbing, not code a fix would live in
NOT_REAL_FIX = re.compile(r"(ForcedRecompilation|BuildInfo|_pb2|\.g\.|generated)", re.I)


def file_at(bare, sha, path, max_lines=400):
    rc, out = git(bare, "show", f"{sha}:{path}")
    if rc:
        return None
    return "\n".join(out.split("\n")[:max_lines])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=0, help="0 = every repo with history")
    ap.add_argument("--per-repo", type=int, default=40)
    ap.add_argument("--max-diff-lines", type=int, default=120)
    ap.add_argument("--out", default=str(DATA_DIR / "episodes/diffs.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    gitdir = os.path.join(ROOT, "data/repos/git")
    clones = json.load(open(ROOT / "data/repos/clones.json"))
    lic = {k: v.get("license") for k, v in clones.items()}
    try:
        for r in json.load(open(ROOT / "data/repos/swesmith_admitted.json")):
            lic.setdefault(r["repo"], r["license"])
    except Exception:
        pass
    bares = sorted(os.listdir(gitdir))
    if a.repos:
        bares = bares[:a.repos]
    st, kept, t0 = Counter(), 0, time.time()
    fh = open(a.out, "w")
    for bn in bares:
        repo = bn[:-4].replace("__", "/")
        bare = os.path.join(gitdir, bn)
        # licence must be one WE admit -- the repo list is wider than the policy
        if lic.get(repo) and lic[repo] not in PERMISSIVE:
            st["repo_licence_rejected"] += 1
            continue
        _rc, logout = git(bare, "log", "--no-merges", "--format=%x01%H%x00%s",
                          "--name-only", "-n2000", timeout=300)
        n_repo = 0
        for b in logout.split("\x01"):
            if n_repo >= a.per_repo:
                break
            ls = [x for x in b.strip().split("\n") if x.strip()]
            if not ls or "\x00" not in ls[0]:
                continue
            sha, subj = ls[0].split("\x00", 1)
            files = ls[1:]
            if not FIX.search(subj) or NOT_FIX.search(subj):
                continue
            if VENDORED.search(" ".join(files)):
                continue
            src = [f for f in files if SRC.search(f) and not TEST.search(f)]
            tst = [f for f in files if TEST.search(f)]
            if len(src) != 1 or not tst:      # single source file: tangling (D4)
                st["not_single_source"] += 1
                continue
            if NOT_REAL_FIX.search(src[0]):
                st["build_plumbing"] += 1
                continue
            _rc, d = git(bare, "show", "--format=", "--unified=3", sha)
            if not d:
                st["no_diff"] += 1
                continue
            td, sd = split_diff(d)
            n_lines = sum(1 for l in sd.split("\n")
                          if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
            if not td.strip() or not sd.strip() or n_lines > a.max_diff_lines:
                st["shape"] += 1
                continue
            tp = (test_files_in(td) or [None])[0]
            body = added_test_body(td)
            if not tp or len(body.strip()) < 60:
                st["thin_test"] += 1
                continue
            before = file_at(bare, sha + "^", src[0])
            if before is None:
                st["no_parent_file"] += 1
                continue
            prompt = PROMPT.format(test_path=tp, test_body=scrub(body),
                                   src_path=src[0], src_before=scrub(before))
            target = scrub(sd)
            fh.write(json.dumps({
                "repo": repo, "commit": sha, "license": lic.get(repo),
                "kind": "commit_diff", "human_target": True,
                "src_path": src[0], "test_path": tp, "n_diff_lines": n_lines,
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": target}],
            }) + "\n")
            kept += 1
            n_repo += 1
            st["kept"] += 1
        if n_repo:
            st["repos_with_tasks"] += 1
        if (bares.index(bn) + 1) % 20 == 0:
            print(f"  [{bares.index(bn) + 1}/{len(bares)}] kept {kept}", flush=True)
    fh.close()
    print(f"\n{kept} diff tasks from {st['repos_with_tasks']} repos "
          f"in {(time.time() - t0) / 60:.1f} min -> {a.out}")
    print(f"drops: {dict(st)}")


if __name__ == "__main__":
    main()
