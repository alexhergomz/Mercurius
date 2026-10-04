"""Capability benchmark on tasks we own, with a repository-disjoint held-out split.

Everything measured so far is perplexity and position-bucketed NLL. Neither can
see tool use, multi-hop retrieval or code understanding -- which is exactly what
the commit-diff and episode tiers were built to buy. So the question "do we need
more data" has been unanswerable, because nothing measured the thing the data is
for.

This is the cheapest honest answer available: the parser oracle that verifies
training tasks also verifies held-out ones, so a benchmark costs nothing but
generation. Properties that matter:

  * WE OWN IT. No licence question, unlike HumanEval/MBPP/GSM8K (usable for
    evaluation, but third-party), and no contamination risk from a public
    benchmark the base model may have memorised.
  * REPOSITORY-DISJOINT. The split is by REPO, not by task: a model that trained
    on other symbols from the same repository has seen its structure, its naming
    conventions and its layout. Splitting by task would leak all of that.
  * It measures the deployed shape -- the model works through the same tools,
    against the same RepoEnv, with the same answer format.

Ceiling caveat: the 27B and 122B teachers both score near 100% on `locate` and
were 12/12 on one repository, so the easy kinds cannot separate models. Reported
per kind for that reason, with `impact` (multi-file, aggregation) as the one that
discriminates.

    python experiments/capability_eval.py --arms base=none pilot=ckpt/adapters-pilotmix-best.pt
"""
import argparse
import json
import os
import random

from mercurius.data.code_graph import CodeGraph, make_tasks, verify
from mercurius.paths import DATA_DIR, ROOT

HELD_OUT = DATA_DIR / "eval" / "held_out_tasks.json"


def build_split(n_repos=10, per_repo=8, seed=17, exclude=()):
    """Tasks from repositories NOT used for training episodes."""
    clones = {k: v for k, v in json.load(open(ROOT / "data/repos/clones.json")).items()
              if "error" not in v}
    yields = json.load(open(ROOT / "data/repos/graph_yield.json"))
    trained = set(exclude)
    for p in ("data/episodes/verified_122b.jsonl", "data/episodes/diffs.jsonl"):
        if os.path.exists(ROOT / p):
            for line in open(ROOT / p):
                trained.add(json.loads(line).get("repo"))
    pool = sorted(r for r, v in yields.items() if v and r not in trained and r in clones)
    random.Random(seed).shuffle(pool)
    out = []
    for repo in pool:
        if len({t["repo"] for t in out}) >= n_repos:
            break
        root = os.path.join(ROOT, clones[repo]["path"])
        if not os.path.isdir(root):
            continue
        try:
            graph = CodeGraph(root)
            tasks = make_tasks(graph, repo, n=per_repo, rng=random.Random(seed))
        except Exception:
            continue
        out.extend(tasks)
    os.makedirs(os.path.dirname(HELD_OUT), exist_ok=True)
    json.dump({"seed": seed, "excluded_repos": sorted(trained), "tasks": out},
              open(HELD_OUT, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=10)
    ap.add_argument("--per-repo", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.rebuild or not os.path.exists(HELD_OUT):
        tasks = build_split(a.repos, a.per_repo)
        print(f"built held-out split: {len(tasks)} tasks over "
              f"{len({t['repo'] for t in tasks})} repositories")
    else:
        tasks = json.load(open(HELD_OUT))["tasks"]
        print(f"loaded {len(tasks)} held-out tasks")
    from collections import Counter
    print("  by kind:", dict(Counter(t["kind"] for t in tasks)))
    print("  by lang:", dict(Counter(t["lang"] for t in tasks)))
    print("  repos  :", sorted({t["repo"] for t in tasks}))
    print(f"\n-> {HELD_OUT}")
    print("verify() is the same oracle used in training, so scoring is mechanical.")


if __name__ == "__main__":
    main()
