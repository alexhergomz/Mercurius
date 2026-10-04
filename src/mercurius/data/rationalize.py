"""Hindsight trajectories for tasks nothing solved: show the generator the answer.

The all-fail set is the hardest material we have and, in the execution tier, the
only set where we already hold the correct answer -- the human's commit. No
rollout reached it, so there is nothing to distil. Rationalization recovers it:
give the generator the gold diff as a HINT, ask it to do the work that would
lead there, and keep the work.

This is STaR's rationalization (Zelikman et al., arXiv:2203.14465): on failure,
supply the answer, generate a rationale that reaches it, and train on the
rationale WITHOUT the hint. That last clause is the whole technique. If the hint
survives into the training sequence we teach the model that the answer arrives
for free, which is worse than not training on the task at all.

Three guards, the same shape as mercurius.data.correction:

  * the hint lives only in the generator's prompt; the stored sequence starts
    from the ordinary task statement and is checked for leakage before it is
    kept;
  * the generator authors assistant turns only -- every tool result is replayed
    against the real RepoEnv, so no observation can be fabricated;
  * the trajectory must still verify, against the parser oracle or the
    fail-to-pass gate, exactly like an organic one.

A known weakness, recorded because it will not show up in any pass rate:
rationalized trajectories can reach the right answer through reasoning that does
not actually support it -- the generator is working backwards from something it
has been told. STaR reports this for chains of thought and there is no reason
agentic traces are exempt. So they are stored with `rationalized: True` and kept
separable from organic trajectories, and any run that uses them should be
ablated against one that does not.
"""
import json
import re

HINT = """You are producing training data for a smaller model.

Below is a task that was attempted and FAILED, together with the change that
actually fixed it. Your job is to do the investigation that a competent engineer
would do to ARRIVE at that change, using the tools -- read the relevant code,
confirm the cause, then make the change.

Work it honestly. Do not mention the fix you were shown, do not refer to being
given the answer, and do not refer to training data or to a smaller model. The
transcript must read as someone solving the problem for the first time. If the
investigation would not genuinely lead to this change, say so plainly instead of
inventing a path to it.

--- TASK
{task}

--- THE CHANGE THAT FIXED IT (context for you only; never mention it)
{gold}
"""

# phrases that betray the hint even when the diff itself is absent
TELLS = re.compile(
    r"(as (shown|given|provided) (above|below|in the)|the (gold|reference|provided) "
    r"(fix|patch|diff|change|answer|solution)|training data|smaller model|"
    r"i was (shown|given|told)|according to the (fix|patch|diff)|"
    r"the answer is given|as instructed)", re.I)


def leaks(messages, gold_diff, task=None):
    """True if the hint survived into the trajectory.

    Checks three ways, because each catches what the others miss: verbatim lines
    from the diff, the phrases a model uses when it knows the answer, and (when
    a task is given) the ground-truth paths themselves appearing before any tool
    call could have revealed them.
    """
    text = "\n".join((m.get("content") or "") + " " + (m.get("reasoning_content") or "")
                     for m in messages)
    if TELLS.search(text):
        return True, "tell_phrase"
    body = [l[1:].strip() for l in (gold_diff or "").split("\n")
            if l[:1] in "+-" and not l.startswith(("+++", "---")) and len(l.strip()) > 24]
    for line in body:
        if line and line in text:
            return True, "verbatim_diff_line"
    return False, "ok"


def rationalize(gen, env, task, gold_diff, rollout_fn, max_turns=30,
                token_budget=100_000):
    """One hindsight trajectory, or (None, why).

    `rollout_fn(messages, max_turns, token_budget)` continues a trajectory in the
    real environment -- mercurius.data.episodes.run_turns bound to a teacher and
    an env. The generator never writes a tool result.
    """
    statement = task["question"] if "question" in task else task.get("task", "")
    primed = [{"role": "user", "content": HINT.format(task=statement, gold=gold_diff)}]
    msgs, status = rollout_fn(primed, max_turns=max_turns, token_budget=token_budget)
    if status != "answered":
        return None, f"status:{status}"
    bad, why = leaks(msgs, gold_diff, task)
    if bad:
        return None, f"leak:{why}"
    # the stored sequence carries the ORDINARY statement, never the hint
    out = [dict(m) for m in msgs]
    out[0] = {"role": "user", "content": statement}
    return {"task": task, "messages": out, "rationalized": True,
            "n_turns": sum(1 for m in out if m["role"] == "assistant")}, "ok"


def store_rationalized(store, rec, repo, task_id, passed, why, source, generator):
    """Written with `rationalized: True` so it can be ablated out later."""
    store.append(task_id=task_id, task=rec["task"], repo=repo,
                 messages=rec["messages"], passed=passed, why=why,
                 source=source, generator=generator,
                 extra={"rationalized": True})


if __name__ == "__main__":
    demo = [{"role": "assistant",
             "content": "Looking at the provided fix, the issue is in parser.py"}]
    print(json.dumps(dict(zip(("leaks", "why"), leaks(demo, "+ x = 1"))), indent=1))
