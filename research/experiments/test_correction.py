"""End-to-end test of correction trajectories on real tasks.

Samples a task until it has BOTH a rejected and an accepted rollout, then asks
the generator to bridge them and checks the result survives every guard:
schema, cut point, answer-leak, real tool replay, and the parser oracle on the
final answer.

    python experiments/test_correction.py --repo jaegertracing/jaeger --tasks 4
"""
import argparse
import functools
import json
import os
import random

from mercurius.data.code_graph import CodeGraph, make_tasks, verify
from mercurius.data.correction import build_correction
from mercurius.data.episodes import Teacher, rollout, run_turns
from mercurius.data.repo_env import RepoEnv
from mercurius.paths import ROOT

SUFFIX = ("\n\nWork it out using the tools; do not guess. When you are certain, "
          "give the answer in the <answer> tags exactly as specified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jaegertracing/jaeger")
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "data/episodes/pilot_corrections.jsonl"))
    a = ap.parse_args()
    clones = json.load(open(ROOT / "data/repos/clones.json"))
    root = os.path.join(ROOT, clones[a.repo]["path"])
    graph = CodeGraph(root)
    env = RepoEnv(root, max_read_lines=1000, search_k=8)
    tasks = make_tasks(graph, a.repo, n=a.tasks, rng=random.Random(7))
    gen = Teacher()
    fh = open(a.out, "a")
    made, tried = 0, 0

    for t in tasks:
        good, bad = [], []
        for _ in range(a.k):
            if good and bad:
                break
            try:
                msgs, status = rollout(gen, env, a.repo, {"task": t["question"] + SUFFIX},
                                       max_turns=30, token_budget=100_000)
            except Exception as err:
                print(f"  rollout failed: {type(err).__name__}")
                continue
            if status != "answered":
                continue
            ok, _ = verify(t, msgs[-1].get("content") or "")
            (good if ok else bad).append(msgs)
        print(f"{t['kind']:<8} {t['symbol'][:30]:<30} accepted={len(good)} rejected={len(bad)}")
        if not (good and bad):
            continue
        tried += 1
        cont = functools.partial(run_turns, gen, env)
        rec, why = build_correction(gen, env, a.repo, t, bad[0], good[0], cont)
        if rec is None:
            print(f"    correction rejected: {why}")
            continue
        made += 1
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"    CORRECTION ok: cut after turn {rec['cut_after_turn']}, "
              f"{rec['n_turns']} turns total")
        print(f"      diagnosis: {rec['diagnosis'][:160]}")
        bridge = [m for m in rec["messages"] if m["role"] == "assistant"][rec["cut_after_turn"] + 1]
        print(f"      bridge   : {(bridge.get('content') or '')[:200]}")
    print(f"\n{made}/{tried} correction trajectories built -> {a.out}")


if __name__ == "__main__":
    main()
