"""Every rollout we ever pay for, kept once and read many ways.

Generation is the expensive step in this project; verification is nearly free.
So nothing generated is discarded, and no downstream decision is baked into
extraction. The store is append-only JSONL, one record per rollout, carrying
enough context to reconstruct any view we later want:

    sft_positive()        accepted trajectories            -- rejection-sampling SFT
    gold_supervision()    tasks NOTHING solved             -- the hard tail, with
                                                              the human fix as target
    verifier_examples()   every rollout + its label        -- V-STaR style verifier
    preference_pairs()    chosen/rejected, same task       -- DPO, if ever wanted
    correction_pairs()    a failure and a success, same task -- Agent-R style bridging

Group composition is recorded, never acted on here, and it means something
different for us than it does in the RL literature. There, an all-pass or
all-fail group has zero advantage and therefore zero gradient, which is why
GRPO/DAPO discard them (arXiv:2504.11343, arXiv:2503.14476). OUR RECOVERY
OBJECTIVE IS DISTILLATION: full-vocabulary teacher and student distributions at
every position, so every sequence gives a dense gradient regardless of outcome
and no group is degenerate.

What the split does tell us:

  * all-pass  -- the teacher finds this easy. Good distillation data; the only
                 reason to stop sampling is budget and near-duplication.
  * all-fail  -- the teacher cannot do it. Its trajectories must NOT be
                 distilled, because that teaches its mistakes. But in the
                 execution tier the task is still valuable: it is the hardest
                 material we have and the human's commit is the correct answer,
                 so it becomes CE supervision on a human target rather than
                 distillation of a model one.
  * mixed     -- difficulty near the teacher's limit; the accepted trajectories
                 are ordinary distillation data.

One bias worth remembering: filtering teacher rollouts by outcome means we
distil a FILTERED teacher, which is better than the teacher itself. That is the
point, but it does mean "excess nats" is measured against sequences selected by
outcome rather than by teacher likelihood.
"""
import json
import os
from collections import Counter, defaultdict

from mercurius.paths import DATA_DIR

STORE = os.path.join(DATA_DIR, "rollouts")


def _path(name):
    os.makedirs(STORE, exist_ok=True)
    return os.path.join(STORE, f"{name}.jsonl")


class RolloutStore:
    """Append-only. Readers group by task_id; nothing is mutated in place."""

    def __init__(self, name="main"):
        self.path = _path(name)
        self._fh = None

    # ------------------------------------------------------------- writing
    def append(self, *, task_id, task, repo, messages, passed, why, source,
               generator, n_turns=None, n_tokens=None, sample_index=0, extra=None):
        """One rollout. `passed` is the VERIFIED outcome, `why` the detail.

        `source` says which oracle judged it ("parser", "fail_to_pass",
        "typecheck"), because their strengths differ and a later reader must be
        able to tell them apart rather than treating all verdicts alike.
        """
        rec = {"task_id": task_id, "repo": repo, "task": task, "passed": bool(passed),
               "why": why, "source": source, "generator": generator,
               "sample_index": sample_index,
               "n_turns": n_turns if n_turns is not None else
               sum(1 for m in messages if m.get("role") == "assistant"),
               "n_tokens": n_tokens, "messages": messages}
        if extra:
            rec.update(extra)
        if self._fh is None:
            self._fh = open(self.path, "a")
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------- reading
    def read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def by_task(self, rows=None):
        g = defaultdict(list)
        for r in (rows if rows is not None else self.read()):
            g[r["task_id"]].append(r)
        return g

    def groups(self, rows=None):
        """task_id -> {'kind': mixed|all_pass|all_fail|single, 'n', 'n_pass', ...}"""
        out = {}
        for tid, rs in self.by_task(rows).items():
            n, npass = len(rs), sum(1 for r in rs if r["passed"])
            kind = ("single" if n < 2 else
                    "all_pass" if npass == n else
                    "all_fail" if npass == 0 else "mixed")
            out[tid] = {"kind": kind, "n": n, "n_pass": npass,
                        "pass_rate": npass / n, "repo": rs[0]["repo"],
                        "source": rs[0]["source"]}
        return out

    # --------------------------------------------------------------- views
    def sft_positive(self, max_per_task=1):
        """Accepted trajectories, capped per task so an easy task cannot flood
        the corpus with near-duplicates."""
        out = []
        for _tid, rs in self.by_task().items():
            good = [r for r in rs if r["passed"]]
            good.sort(key=lambda r: (r["n_turns"] or 0))     # prefer the direct one
            out.extend(good[:max_per_task])
        return out

    def gold_supervision(self):
        """Tasks NOTHING solved -- the teacher's ceiling.

        These must not be distilled: every trajectory in them is the teacher
        being wrong, and distillation would copy that. In the execution tier
        they are still the most valuable material we have, because the human's
        commit is the correct answer -- so they belong in the CE data term
        against a HUMAN target, not in the divergence term against a model one.
        """
        g = self.groups()
        return [tid for tid, m in g.items() if m["kind"] == "all_fail"]

    def verifier_examples(self, balance=True):
        """(trajectory, label) for a best-of-k verifier. SWE-Gym balanced ~1318
        successes against ~1318 failures; unbalanced data teaches the prior, not
        the discrimination."""
        rows = self.read()
        pos = [r for r in rows if r["passed"]]
        neg = [r for r in rows if not r["passed"]]
        if balance:
            k = min(len(pos), len(neg))
            pos, neg = pos[:k], neg[:k]
        return pos + neg

    def preference_pairs(self, max_per_task=1, length_controlled=True):
        """(chosen, rejected) from the same task. Built on demand and NOT used
        today -- on verifiable rewards, rejection-sampling SFT beats DPO
        (52.3 vs 48.8, arXiv:2504.11343). Kept because the pairs are free once
        the rollouts exist.

        length_controlled pairs the chosen with the rejected CLOSEST in length:
        DPO's margin can be inflated by length alone, a property of the
        contrastive loss rather than of the data (arXiv:2403.19159).
        """
        out = []
        for _tid, rs in self.by_task().items():
            good = [r for r in rs if r["passed"]]
            bad = [r for r in rs if not r["passed"]]
            if not good or not bad:
                continue
            for c in good[:max_per_task]:
                cl = c.get("n_tokens") or 0
                r = (min(bad, key=lambda x: abs((x.get("n_tokens") or 0) - cl))
                     if length_controlled else bad[0])
                out.append({"task_id": c["task_id"], "repo": c["repo"],
                            "task": c["task"], "chosen": c["messages"],
                            "rejected": r["messages"],
                            "len_delta": abs((r.get("n_tokens") or 0) - cl)})
        return out

    def correction_pairs(self):
        """Tasks holding BOTH a failure and a success -- the input an Agent-R
        style bridge needs (mercurius.data.correction)."""
        out = []
        for tid, rs in self.by_task().items():
            good = [r for r in rs if r["passed"]]
            bad = [r for r in rs if not r["passed"]]
            if good and bad:
                out.append({"task_id": tid, "task": good[0]["task"],
                            "repo": good[0]["repo"],
                            "accepted": good[0]["messages"],
                            "rejected": bad[0]["messages"]})
        return out

    def summary(self):
        rows = self.read()
        g = self.groups(rows)
        return {"rollouts": len(rows), "tasks": len(g),
                "passed": sum(1 for r in rows if r["passed"]),
                "groups": dict(Counter(m["kind"] for m in g.values())),
                "by_source": dict(Counter(r["source"] for r in rows)),
                "by_generator": dict(Counter(r["generator"] for r in rows)),
                "pairs_available": len(self.correction_pairs())}


if __name__ == "__main__":
    import sys
    s = RolloutStore(sys.argv[1] if len(sys.argv) > 1 else "main")
    print(json.dumps(s.summary(), indent=1))
