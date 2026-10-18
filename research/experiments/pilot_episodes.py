"""Pilot batch of repository agent episodes, to measure quality and cost.

Runs several episodes concurrently against llama-server (the teacher decodes
at ~11 tok/s per slot, so concurrency is what fills the 4 slots). Reports the
filter reasons, the token-length distribution of the rendered sequences, and
the cost per kept episode -- the numbers that decide the mixture's size and
how the training loader should pack them.

    python experiments/pilot_episodes.py --repos 20 --tasks 3 --workers 6
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import threading
import time

from transformers import AutoTokenizer

from mercurius.data.episodes import Teacher, keep, make_tasks, rollout
from mercurius.data.repo_env import TOOLS, RepoEnv
from mercurius.paths import DATA_DIR, ROOT, STAGE_AB

LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=20)
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(DATA_DIR / "episodes/pilot.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    clones = {k: v for k, v in json.load(open(ROOT / "data/repos/clones.json")).items()
              if "error" not in v}
    rng = random.Random(a.seed)
    names = rng.sample(sorted(clones), min(a.repos, len(clones)))
    teacher = Teacher()
    stats, lens, t0 = {}, [], time.time()
    fh = open(a.out, "a")

    def one(name):
        meta = clones[name]
        env = RepoEnv(os.path.join(ROOT, meta["path"]))
        out = []
        try:
            tasks = make_tasks(teacher, env, name, n=a.tasks, rng=random.Random(hash(name) & 0xffff))
        except Exception as err:
            return [("task_gen_failed", None, str(err)[:80])]
        for t in tasks[:a.tasks]:
            try:
                msgs, status = rollout(teacher, env, name, t)
                ok, why = keep(msgs, status, t)
            except Exception as err:
                out.append(("rollout_failed", None, str(err)[:80]))
                continue
            n_tok = None
            if ok:
                s = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
                n_tok = len(tok(s).input_ids)
                rec = {"repo": name, "commit": meta.get("commit"), "license": meta.get("license"),
                       "task": t, "tools": TOOLS, "messages": msgs, "n_tokens": n_tok,
                       "n_turns": sum(1 for m in msgs if m["role"] == "assistant")}
                with LOCK:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
            out.append((why, n_tok, None))
        return out

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(one, names)):
            for why, n_tok, err in res:
                stats[why] = stats.get(why, 0) + 1
                if n_tok:
                    lens.append(n_tok)
                if err:
                    print(f"    {why}: {err}", flush=True)
            el = time.time() - t0
            print(f"  [{i + 1}/{len(names)}] {el / 60:5.1f} min  kept {len(lens)}  {stats}", flush=True)
    el = time.time() - t0
    lens.sort()
    if lens:
        q = lambda p: lens[int(p * (len(lens) - 1))]
        print(f"\nkept {len(lens)} episodes in {el / 60:.1f} min "
              f"({el / max(len(lens), 1):.0f} s each, {a.workers} workers)")
        print(f"tokens per episode: min {lens[0]}  p25 {q(.25)}  median {q(.5)}  "
              f"p75 {q(.75)}  max {lens[-1]}  total {sum(lens) / 1e6:.2f} M")
        print(f"share >= 8192 tokens: {sum(l >= 8192 for l in lens) / len(lens):.0%}")
    print(f"filters: {stats}\n-> {a.out}")


if __name__ == "__main__":
    main()
