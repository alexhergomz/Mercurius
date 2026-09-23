"""Build agent episodes over every admitted repository clone.

Same pipeline as the pilot (experiments/pilot_episodes.py) with the tuned
generation economics: bigger tool results and a higher turn cap, so each
generated decision buys more real repository text (docs/data_policy.md). The
teacher's decode is the bottleneck, not the repositories.

Resumable: repositories already present in the output file are skipped, so the
build can be stopped and restarted.

    python scripts/build_episodes.py --tasks 6 --workers 6 --target-tokens 4000000
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
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--read-lines", type=int, default=1000)
    ap.add_argument("--search-k", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--token-budget", type=int, default=100_000,
                    help="wind the episode up before this many tokens. The "
                         "per-slot context is 262k (the model's full window), "
                         "so this is about what training can USE: an episode "
                         "longer than --seq keeps only its first --seq tokens.")
    ap.add_argument("--target-tokens", type=int, default=4_000_000)
    ap.add_argument("--out", default=str(DATA_DIR / "episodes/build1.jsonl"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    clones = {k: v for k, v in json.load(open(ROOT / "data/repos/clones.json")).items()
              if "error" not in v}
    done_repos, done_tok = set(), 0
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done_repos.add(r["repo"]); done_tok += r.get("n_tokens", 0)
        print(f"resuming: {len(done_repos)} repositories, {done_tok / 1e6:.2f} M tokens already",
              flush=True)
    # biggest repositories first: longer files and search results per call
    names = sorted((n for n in clones if n not in done_repos),
                   key=lambda n: -os.path.getsize(os.path.join(ROOT, clones[n]["path"]))
                   if os.path.isdir(os.path.join(ROOT, clones[n]["path"])) else 0)
    teacher = Teacher()
    stats, total, t0 = {}, done_tok, time.time()
    stop = threading.Event()
    fh = open(a.out, "a")

    def one(name):
        if stop.is_set():
            return []
        meta = clones[name]
        env = RepoEnv(os.path.join(ROOT, meta["path"]), max_read_lines=a.read_lines,
                      search_k=a.search_k)
        out = []
        try:
            tasks = make_tasks(teacher, env, name, n=a.tasks,
                               rng=random.Random(a.seed ^ hash(name) & 0xffff))
        except Exception as err:
            return [("task_gen_failed", 0)]
        for t in tasks[:a.tasks]:
            if stop.is_set():
                break
            try:
                msgs, status = rollout(teacher, env, name, t, max_turns=a.max_turns,
                                       token_budget=a.token_budget)
                ok, why = keep(msgs, status, t)
            except Exception:
                out.append(("rollout_failed", 0)); continue
            n_tok = 0
            if ok:
                s = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False)
                n_tok = len(tok(s).input_ids)
                rec = {"repo": name, "commit": meta.get("commit"), "license": meta.get("license"),
                       "task": t, "tools": TOOLS, "messages": msgs, "n_tokens": n_tok,
                       "n_turns": sum(1 for m in msgs if m["role"] == "assistant")}
                with LOCK:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
            out.append((why, n_tok))
        return out

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(one, names)):
            for why, n_tok in res:
                stats[why] = stats.get(why, 0) + 1
                total += n_tok
            el = (time.time() - t0) / 3600
            print(f"  [{i + 1}/{len(names)}] {total / 1e6:.2f} M tokens, "
                  f"{el:.2f} h, {(total - done_tok) / max(el, 1e-9) / 1e6:.2f} M/h  {stats}",
                  flush=True)
            if total >= a.target_tokens:
                print("  target reached, finishing", flush=True)
                stop.set()
                break
    print(f"total {total / 1e6:.2f} M tokens in {(time.time() - t0) / 3600:.2f} h; {stats}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
