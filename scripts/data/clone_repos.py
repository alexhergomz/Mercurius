"""Shallow-clone a language-stratified batch of admitted repositories.

Admitted = permissive license (scripts/repo_licenses.py), not a SWE-bench
Verified repository, not opted out (scripts/stack_optouts.py). Each clone is
pinned: the commit SHA is recorded, so an episode built on it can be traced
and rebuilt. A clone whose working tree has no license file is discarded even
if GitHub detected a license -- the file is what grants the rights.

    python scripts/clone_repos.py --n 150      # -> data/repos/src/, data/repos/clones.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import subprocess

ROOT = os.path.join(os.path.dirname(__file__), "..")
LIC = os.path.join(ROOT, "data/repos/licenses.json")
SRC = os.path.join(ROOT, "data/repos/src")
OUT = os.path.join(ROOT, "data/repos/clones.json")
LICENSE_NAMES = ("license", "licence", "copying", "unlicense")


def has_license_file(path):
    return any(f.lower().split(".")[0] in LICENSE_NAMES for f in os.listdir(path))


def clone(name, meta):
    dst = os.path.join(SRC, name.replace("/", "__"))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    r = subprocess.run(["git", "clone", "--depth", "1", "--quiet",
                        f"https://github.com/{meta.get('full_name', name)}.git", dst],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return name, {"error": r.stderr.strip()[-200:]}
    if not has_license_file(dst):
        shutil.rmtree(dst)
        return name, {"error": "no license file in tree"}
    sha = subprocess.run(["git", "-C", dst, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    shutil.rmtree(os.path.join(dst, ".git"), ignore_errors=True)   # content only
    return name, {"path": os.path.relpath(dst, ROOT), "commit": sha,
                  "license": meta["license"], "language": meta.get("language")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-stars", type=int, default=20)
    ap.add_argument("--max-size-mb", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lic = json.load(open(LIC))
    pool = [(k, m) for k, m in lic.items()
            if m.get("admitted") and m.get("stars", 0) >= a.min_stars
            and m.get("size_kb", 0) <= a.max_size_mb * 1024 and not m.get("fork")]
    by_lang = {}
    for k, m in pool:
        by_lang.setdefault(m.get("language") or "other", []).append((k, m))
    rng = random.Random(a.seed)
    for v in by_lang.values():
        rng.shuffle(v)
    pick, langs = [], sorted(by_lang, key=lambda l: -len(by_lang[l]))
    while len(pick) < a.n and any(by_lang.values()):          # round-robin over languages
        for l in langs:
            if by_lang[l] and len(pick) < a.n:
                pick.append(by_lang[l].pop())
    os.makedirs(SRC, exist_ok=True)
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [(k, m) for k, m in pick if k not in done or "error" in done[k]]
    print(f"pool {len(pool)} admitted repos; cloning {len(todo)} "
          f"({len({(m.get('language') or 'other') for _, m in pick})} languages)", flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        for name, res in ex.map(lambda km: clone(*km), todo):
            done[name] = res
    json.dump(done, open(OUT, "w"), indent=0)
    ok = [v for v in done.values() if "error" not in v]
    from collections import Counter
    print(f"cloned {len(ok)}; failed {len(done) - len(ok)}; "
          f"languages {Counter(v['language'] for v in ok).most_common(10)}")


if __name__ == "__main__":
    main()
