"""Correction trajectories: a failed attempt, a realisation, and a real recovery.

Given a REJECTED and an ACCEPTED rollout of the same verified task, the
generator writes the turn that joins them -- the moment the agent notices it is
on the wrong track and changes approach. The result is one trajectory containing
real exploration, a real wrong turn, a synthesised realisation, and a real
recovery ending in a parser-verified answer.

Why the generator picks the cut point rather than a heuristic: there is no
mechanical definition of "the first wrong action". Opening a file that turns out
to be irrelevant is not a mistake, it is search -- agents rule things out. A
heuristic that flags the first tool call which missed a ground-truth file would
mark healthy exploration as the error and teach the realisation at a point where
nothing had gone wrong, a bias that would be invisible because the trajectories
would still verify. It also imposes one canonical shape (exactly one wrong turn,
always in the middle) when real failures often come from a bad premise in the
first turn or from stopping too early in the last.

Nothing is taken on trust. The bridge is constrained by a JSON schema, and then:

  * every tool call after the cut is REPLAYED against the real RepoEnv, so a
    fabricated observation cannot survive -- the generator never authors a tool
    result, it only authors assistant turns;
  * the bridge must not contain the answer, checked against the parsed ground
    truth, so the recovery has to actually find it rather than assert it;
  * the finished trajectory is verified by the same parser oracle as any other.

Related work: Agent-R (arXiv:2501.11425) splices failed into successful
trajectories at the divergence point to teach recovery; LEMA (arXiv:2310.20689)
has a strong model write mistake-correction pairs. Both cited from memory and
worth re-checking before leaning on them.
"""
import json
import re

from mercurius.data.code_graph import verify
from mercurius.data.repo_env import TOOLS

BRIDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "cut_after_turn": {
            "type": "integer",
            "description": "Index (0-based, counting ASSISTANT turns only) of the "
                           "last assistant turn to KEEP from the failed attempt. "
                           "Everything after it is discarded.",
        },
        "diagnosis": {
            "type": "string",
            "description": "One sentence: what the agent had got wrong by that point.",
        },
        "reasoning": {
            "type": "string",
            "description": "The agent's private reasoning at the moment of realising, "
                           "written in first person as the agent would think it.",
        },
        "content": {
            "type": "string",
            "description": "What the agent says next: notices the approach is not "
                           "working, and states what it will do instead. Natural "
                           "in-character text -- never mention being taught, a "
                           "student, a teacher, or a corrected trajectory.",
        },
        "tool_calls": {
            "type": "array",
            "description": "The corrective next action(s). Use the same tools.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["cut_after_turn", "diagnosis", "reasoning", "content", "tool_calls"],
}

INSTRUCTION = """You are building training data that teaches a smaller model to \
recover when it goes down the wrong path.

Below are two attempts at the same repository question: one that FAILED and one \
that SUCCEEDED. Your job is to write the single turn that joins them -- the \
moment in the failed attempt where the agent should have realised it was on the \
wrong track, and what it does instead.

Choose `cut_after_turn` yourself. Cut where the attempt actually went wrong, \
which may be early (a bad premise), in the middle (a wrong file), or late \
(concluding too soon). Do not cut at a point where the agent was still \
legitimately searching -- ruling out a file is not a mistake.

Write the bridge IN CHARACTER, as the agent itself, at that moment. It must read \
as a genuine realisation: notice the problem, say what to do instead, and take \
the corrective action. Never refer to being taught, to a student, to a teacher, \
or to this being a constructed example.

Do NOT state the answer. The bridge changes the APPROACH; the agent must still \
find the answer with the tools afterwards.

--- THE QUESTION
{question}

--- FAILED ATTEMPT (assistant turns are numbered)
{failed}

--- SUCCESSFUL ATTEMPT (for your reference: how it was actually solved)
{ok}

Reply with JSON only, matching this schema:
{schema}"""


def _render(msgs, number=False):
    out, n = [], 0
    for m in msgs:
        if m["role"] == "assistant":
            tag = f"[assistant turn {n}]" if number else "[assistant]"
            n += 1
            body = m.get("content") or ""
            if m.get("tool_calls"):
                calls = "; ".join(f"{c['function']['name']}({json.dumps(c['function']['arguments'])[:200]})"
                                  for c in m["tool_calls"])
                body = (body + f"\n  -> calls: {calls}").strip()
            out.append(f"{tag} {body[:900]}")
        elif m["role"] == "tool":
            out.append(f"[tool result] {(m.get('content') or '')[:400]}")
        elif m["role"] == "user" and out:
            out.append(f"[user] {(m.get('content') or '')[:200]}")
    return "\n".join(out)


def _assistant_indices(msgs):
    return [i for i, m in enumerate(msgs) if m["role"] == "assistant"]


def leaks_answer(text, task):
    """The bridge must not hand over the answer it is supposed to make the agent
    go and find."""
    ans = task["answer"]
    paths = list(ans.get("files", [])) + ([ans["file"]] if "file" in ans else [])
    low = (text or "").lower()
    return any(p.lower() in low for p in paths if p)


def synthesize_bridge(gen, task, rejected, accepted, max_tokens=1200):
    prompt = INSTRUCTION.format(
        question=task["question"], failed=_render(rejected, number=True),
        ok=_render(accepted), schema=json.dumps(BRIDGE_SCHEMA, indent=1))
    out = gen.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens,
                   temperature=0.4, think=False)
    txt = out.get("content") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None, "no_json"
    try:
        b = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "bad_json"
    for k in BRIDGE_SCHEMA["required"]:
        if k not in b:
            return None, f"missing:{k}"
    return b, "ok"


def build_correction(gen, env, repo, task, rejected, accepted, rollout_fn,
                     max_turns=20, token_budget=100_000):
    """One correction trajectory, or (None, why).

    The generator authors only assistant turns. Every tool result comes from the
    real environment, here and in the continuation.
    """
    bridge, why = synthesize_bridge(gen, task, rejected, accepted)
    if bridge is None:
        return None, why
    idx = _assistant_indices(rejected)
    cut = bridge.get("cut_after_turn")
    if not isinstance(cut, int) or not (0 <= cut < len(idx)):
        return None, f"bad_cut:{cut}"
    if leaks_answer(bridge.get("content", "") + " " + bridge.get("reasoning", ""), task):
        return None, "bridge_leaks_answer"
    calls = bridge.get("tool_calls") or []
    if not calls:
        return None, "no_corrective_action"

    # prefix: up to and including the kept assistant turn, plus its tool results
    end = idx[cut] + 1
    while end < len(rejected) and rejected[end]["role"] == "tool":
        end += 1
    msgs = [dict(m) for m in rejected[:end]]

    turn = {"role": "assistant", "content": bridge["content"],
            "reasoning_content": bridge["reasoning"],
            "tool_calls": [{"type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}}
                           for c in calls]}
    msgs.append(turn)
    for c in turn["tool_calls"]:                      # REAL results, never authored
        msgs.append({"role": "tool", "content": env.call(c["function"]["name"],
                                                          c["function"]["arguments"])})
    if any((m.get("content") or "").startswith("Error: unknown tool") for m in msgs[-len(calls):]):
        return None, "bridge_bad_tool"

    tail, status = rollout_fn(msgs, max_turns=max_turns, token_budget=token_budget)
    if status != "answered":
        return None, f"continuation:{status}"
    ok, vwhy = verify(task, tail[-1].get("content") or "")
    if not ok:
        return None, f"continuation_wrong:{vwhy}"
    return {"repo": repo, "task": task, "kind": "correction", "verified": True,
            "cut_after_turn": cut, "diagnosis": bridge["diagnosis"],
            "tools": TOOLS, "messages": tail,
            "n_turns": sum(1 for m in tail if m["role"] == "assistant")}, "ok"
