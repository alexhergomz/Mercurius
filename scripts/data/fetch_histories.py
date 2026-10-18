"""Add git history to the admitted clones, as separate treeless bare repos.

The clones were made with `--depth 1` and no `.git`, which was right for a
tool-environment corpus and wrong for everything else: a repository's history is
the only large source of VERIFIED, REAL software-engineering work we have. Every
bug-fix commit is a real defect, located exactly, by the people who owned the
code -- ground truth no static analysis and no model produced.

Stored as bare treeless clones (`--filter=blob:none`) under data/repos/git/ so
the working trees the agent explores are untouched. Treeless is cheap to make
and free to traverse: Jaeger is 10 MB for 4,815 commits, and `git log
--name-only` over 4,000 of them takes 0.2 s because trees ship with the clone.
Only file CONTENTS are fetched on demand, which matters just when a task needs
a parent checkout.

    python scripts/fetch_histories.py --workers 8
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time

from mercurius.paths import ROOT

OUT = os.path.join(ROOT, "data/repos/git")


def fetch(name):
    dest = os.path.join(OUT, name.replace("/", "__") + ".git")
    if os.path.isdir(dest):
        return name, "exists", 0
    t0 = time.time()
    r = subprocess.run(["git", "clone", "--filter=blob:none", "--bare", "--quiet",
                        f"https://github.com/{name}.git", dest],
                       capture_output=True, text=True, timeout=900)
    if r.returncode:
        return name, f"failed: {r.stderr.strip()[:80]}", time.time() - t0
    n = subprocess.run(["git", "-C", dest, "rev-list", "--count", "HEAD"],
                       capture_output=True, text=True)
    return name, f"ok {n.stdout.strip()} commits", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-yielding", action="store_true", default=True,
                    help="only repositories that yield graph tasks")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    clones = {k: v for k, v in json.load(open(ROOT / "data/repos/clones.json")).items()
              if "error" not in v}
    names = sorted(clones)
    yp = ROOT / "data/repos/graph_yield.json"
    if a.only_yielding and yp.exists():
        y = json.load(open(yp))
        names = [n for n in names if y.get(n)]
    print(f"fetching history for {len(names)} repositories", flush=True)
    ok = fail = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, (name, status, el) in enumerate(ex.map(fetch, names)):
            if status.startswith("failed"):
                fail += 1
            else:
                ok += 1
            if (i + 1) % 10 == 0 or status.startswith("failed"):
                print(f"  [{i + 1}/{len(names)}] {name}: {status} ({el:.0f}s)", flush=True)
    sz = subprocess.run(["du", "-sh", OUT], capture_output=True, text=True).stdout.split()[0]
    print(f"done: {ok} ok, {fail} failed, {sz} in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
