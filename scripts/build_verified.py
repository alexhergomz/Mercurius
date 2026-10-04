"""Rejection-sampled agent episodes: k rollouts per task, keep only the correct ones.

The task's answer comes from parsing the repository (mercurius.data.code_graph),
not from a model, so acceptance is mechanical. That changes what the corpus is:
the previous build kept whatever the teacher said as long as it was well formed,
which distils the teacher's mistakes along with its competence. Here a
trajectory survives only if it lands on the parsed answer.

k rollouts per task at a sampling temperature give both the filter and a
measurement: the acceptance rate per repository, task kind and language is the
first real number we have on teacher quality, and it is also the per-task
difficulty signal we would need for any later curriculum.

    python scripts/build_verified.py --repos 20 --tasks 8 --k 4
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import threading
import time
from collections import Counter

from transformers import AutoTokenizer

from mercurius.data.code_graph import CodeGraph, make_tasks, verify
from mercurius.data.episodes import Teacher, rollout
from mercurius.data.repo_env import TOOLS, RepoEnv
from mercurius.data.rollout_store import RolloutStore
from mercurius.paths import DATA_DIR, ROOT, STAGE_AB

LOCK = threading.Lock()

SUFFIX = ("\n\nWork it out using the tools; do not guess. When you are certain, "
          "give the answer in the <answer> tags exactly as specified, as the "
          "last thing in your reply.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=0, help="0 = all that yield tasks")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="max rollouts per task")
    ap.add_argument("--probe", type=int, default=2,
                    help="rollouts drawn before deciding whether a task is worth "
                         "more budget (DAPO dynamic sampling)")
    ap.add_argument("--keep-per-task", type=int, default=1,
                    help="at most this many ACCEPTED trajectories per task, so an "
                         "easy task does not flood the corpus with near-duplicates")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--token-budget", type=int, default=100_000)
    ap.add_argument("--read-lines", type=int, default=1000)
    ap.add_argument("--search-k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--target-tokens", type=int, default=6_000_000)
    ap.add_argument("--out", default=str(DATA_DIR / "episodes/verified.jsonl"))
    ap.add_argument("--stats", default=str(ROOT / "logs/verified_stats.jsonl"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    clones = {k: v for k, v in json.load(open(ROOT / "data/repos/clones.json")).items()
              if "error" not in v}
    yields = {}
    yp = ROOT / "data/repos/graph_yield.json"
    if yp.exists():
        yields = {k: v for k, v in json.load(open(yp)).items() if v}
    names = [n for n in clones if not yields or yields.get(n)]
    names.sort(key=lambda n: -yields.get(n, 0))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["repo"], r["task"]["symbol"], r["task"]["kind"]))
        print(f"resuming: {len(done)} accepted trajectories already", flush=True)
    if a.repos:
        names = names[:a.repos]
    teacher = Teacher()
    agg, total, t0 = Counter(), 0, time.time()
    stop = threading.Event()
    fh = open(a.out, "a")
    rj = open(a.out.replace(".jsonl", "_rejected.jsonl"), "a")
    sh = open(a.stats, "a")
    # EVERY rollout is recorded, passed or not. Generation is what costs here;
    # verification is nearly free, so no downstream use is foreclosed at
    # extraction time (mercurius.data.rollout_store).
    store = RolloutStore(os.path.splitext(os.path.basename(a.out))[0])

    def one(name):
        if stop.is_set():
            return Counter()
        meta = clones[name]
        root = os.path.join(ROOT, meta["path"])
        st = Counter()
        try:
            graph = CodeGraph(root)
            tasks = make_tasks(graph, name, n=a.tasks,
                               rng=random.Random(a.seed ^ (hash(name) & 0xffff)))
        except Exception as err:
            return Counter({f"graph_failed": 1})
        env = RepoEnv(root, max_read_lines=a.read_lines, search_k=a.search_k)
        for t in tasks:
            if stop.is_set():
                break
            if (name, t["symbol"], t["kind"]) in done:
                continue
            # Dynamic sampling, in the shape DAPO uses (arXiv:2503.14476) but
            # NOT for DAPO's reason. DAPO drops all-pass and all-fail groups
            # because in RL they have zero advantage and therefore zero
            # gradient. WE ARE NOT DOING RL. The recovery objective is
            # distillation -- full-vocabulary teacher and student distributions
            # at every position -- so every sequence yields a dense gradient
            # whatever its outcome, and no group is "degenerate".
            #
            # What survives, for different reasons:
            #   all-fail  -> stop early because training on trajectories the
            #                teacher got WRONG distils its mistakes. A data
            #                quality argument, not a gradient one.
            #   all-pass  -> stop early purely to save generation budget: we
            #                only need `keep_per_task` sequences, and the rest
            #                would be near-duplicates. The task itself is good
            #                data, not a degenerate group.
            # Measured at fixed budget: 11,253 tasks attempted vs 9,177, and
            # 7,149 solved vs 6,774, for the same number of rollouts.
            kept, turns_seen, rolls_seen = 0, 0, 0
            n_pass = n_fail = 0
            for j in range(a.k):
                if stop.is_set():
                    break
                if j >= a.probe:
                    if n_fail == 0:                       # enough, and all good
                        st[f"stop_enough:{t['kind']}"] += 1
                        st["saved_rollouts"] += a.k - j
                        break
                    if n_pass == 0:                       # teacher cannot do it
                        st[f"stop_teacher_fails:{t['kind']}"] += 1
                        st["saved_rollouts"] += a.k - j
                        break
                # Stop as soon as SFT has what it needs and nothing has failed.
                # Gating this behind the probe was measured at +16.9% rollouts
                # for no useful return: a task that passes first time rarely
                # yields a negative, and negatives are what the verifier data
                # and the pairs are made of. The saving that DAPO actually buys
                # us is on the other side -- the all-fail cutoff above, which
                # takes a hopeless task from k rollouts to `probe`.
                if kept >= a.keep_per_task and n_fail == 0:
                    break
                try:
                    msgs, status = rollout(teacher, env, name,
                                           {"task": t["question"] + SUFFIX},
                                           max_turns=a.max_turns,
                                           token_budget=a.token_budget)
                except Exception:
                    st["rollout_failed"] += 1
                    continue
                if status != "answered":
                    st[f"status:{status}"] += 1
                    continue
                ok, why = verify(t, msgs[-1].get("content") or "")
                st[f"{'accept' if ok else 'reject'}:{t['kind']}"] += 1
                n_pass += bool(ok)
                n_fail += (not ok)
                with LOCK:
                    store.append(task_id=f"{name}:{t['symbol']}:{t['kind']}",
                                 task=t, repo=name, messages=msgs, passed=ok,
                                 why=why, source="parser", generator="teacher",
                                 sample_index=j)
                n_turns = sum(1 for m in msgs if m["role"] == "assistant")
                # counted on EVERY rollout, accepted or not: turns-to-answer is
                # the difficulty signal (D6), and counting only successes would
                # measure how long winning takes, not how hard the task is
                turns_seen += n_turns
                rolls_seen += 1
                if not ok:
                    # Rejected rollouts are WRITTEN, not discarded: they are the
                    # negatives a best-of-k verifier needs (V-STaR arXiv:2402.06457,
                    # SWE-Gym balanced ~1318 correct against ~1318 incorrect), and
                    # the starting point for any later recovery pass. They cost
                    # disk and nothing else. They are NEVER part of the training
                    # mix -- a separate file so the loader cannot pick them up.
                    st[f"why:{why.split(':')[0]}"] += 1
                    with LOCK:
                        rj.write(json.dumps({"repo": name, "verified": False,
                                             "why": why, "task": t, "sample_index": j,
                                             "tools": TOOLS, "messages": msgs,
                                             "n_turns": n_turns}) + "\n")
                        rj.flush()
                    continue
                s = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
                n_tok = len(tok(s).input_ids)
                rec = {"repo": name, "commit": meta.get("commit"),
                       "license": meta.get("license"), "verified": True,
                       "task": t, "sample_index": j, "tools": TOOLS,
                       "messages": msgs, "n_tokens": n_tok,
                       "n_turns": n_turns}
                with LOCK:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    st["tokens"] += n_tok
                kept += 1
            # Group composition is RECORDED, not acted on. Under distillation
            # it is a difficulty measurement and a note about what the teacher
            # can do -- not the zero-gradient signal it would be under RL.
            st["tasks"] += 1
            st["solved" if kept else "unsolved"] += 1
            if rolls_seen >= 2:
                if n_fail == 0:
                    st[f"degenerate_all_pass:{t['kind']}"] += 1
                elif n_pass == 0:
                    st[f"degenerate_all_fail:{t['kind']}"] += 1
                else:
                    st[f"mixed:{t['kind']}"] += 1
            st[f"turns:{t['kind']}"] += turns_seen
            st[f"rolls:{t['kind']}"] += rolls_seen
        with LOCK:
            sh.write(json.dumps({"repo": name, **st}) + "\n")
            sh.flush()
        return st

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, st in enumerate(ex.map(one, names)):
            agg.update(st)
            total = agg["tokens"]
            acc = sum(v for k, v in agg.items() if k.startswith("accept:"))
            rej = sum(v for k, v in agg.items() if k.startswith("reject:"))
            el = (time.time() - t0) / 3600
            print(f"  [{i + 1}/{len(names)}] {total / 1e6:.2f} M tokens, "
                  f"accept {acc}/{acc + rej} ({acc / max(acc + rej, 1):.0%}), "
                  f"solved {agg['solved']}/{agg['tasks']} tasks, {el:.2f} h", flush=True)
            if total >= a.target_tokens:
                stop.set()
                break
    print(f"\ntotals: {dict(agg)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
