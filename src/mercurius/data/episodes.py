"""Repository agent episodes: the teacher works real tasks with real tools.

For each admitted repository clone (scripts/clone_repos.py):

  1. TASKS. The teacher reads the repository's own README, its file tree and
     one sampled source file, and writes realistic developer tasks about it --
     where is X implemented, how does Y work, what would change to support Z,
     what does test T check. No GitHub issue text is used: issues are not
     covered by the repository's license (docs/data_policy.md). Where a task
     has a checkable answer (a file path, a symbol) it is recorded.
  2. ROLLOUT. The teacher works each task as a deployed agent would: the chat
     carries the real tool schemas (repo_env.TOOLS), every tool call is
     executed against the checkout, and the (scrubbed) result goes back as the
     tool response. The teacher's thinking is kept as reasoning_content, so
     the rendered sequence has the <think> blocks the student will produce.
  3. FILTER. Episodes are kept only if they used at least one tool, ended with
     a final answer, did not loop (the same call three times), stayed within
     the turn budget, and -- where a checkable answer exists -- mention it.

Output: one JSON line per episode with messages, tools, repository, commit and
license, rendered later through the student's chat template. The long parts of
an episode are tool results (real repository text, prefill-only); the teacher
generates only the short decisions, which is what makes this affordable.

The teacher is served by llama-server (OpenAI-compatible API), e.g. the 27B at
4-bit GGUF with an 8-bit KV cache.
"""
import json
import os
import random
import re
import time

import requests

from mercurius.data.repo_env import TOOLS, RepoEnv, scrub

SYSTEM = ("You are an expert software engineering agent working inside the "
          "repository `{repo}`. You can only see the repository through the "
          "provided tools. Investigate before answering, cite file paths and "
          "line numbers, and give a precise final answer when you are done. "
          "Think briefly -- a few sentences at most -- before each tool call.")

TASK_PROMPT = """You are writing realistic tasks for a software engineering agent that will work inside the repository `{repo}` using tools to list directories, read files, grep and search code.

README (truncated):
{readme}

File tree (truncated):
{tree}

One source file, `{path}`:
{code}

Write {n} diverse, specific tasks a developer might actually ask about THIS repository. Mix: locating where something is implemented, explaining how a mechanism works end to end, planning a concrete change (which files and functions to modify and how), and understanding what a test checks. Each task must be answerable by exploring the repository. Where the answer includes a specific file path, give it as "answer_path" (repository-relative); otherwise use null.

Reply with ONLY a JSON list: [{{"task": "...", "kind": "locate|explain|plan|test", "answer_path": "... or null"}}]"""


class Teacher:
    def __init__(self, url="http://127.0.0.1:8080/v1", model="teacher", timeout=600):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def chat(self, messages, tools=None, max_tokens=800, temperature=0.6,
             think=True):
        """think=False disables the model's <think> block for this call.

        Thinking is kept for ROLLOUTS -- the student should learn to think
        before a tool call, and with tools in context the teacher's reasoning
        is short (measured: ~87 generated tokens per turn). It is disabled for
        TASK GENERATION, an open-ended prompt where the teacher spent its whole
        budget reasoning: 105 s and a truncated answer with thinking on
        against 16 s and valid JSON with it off.
        """
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": 0.95}
        if tools:
            body["tools"] = tools
        if not think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = requests.post(f"{self.url}/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


def _tree(env, max_lines=120):
    dirs = sorted({os.path.dirname(f) for f in env.files if f.count(os.sep) <= 2})
    lines = [d + "/" for d in dirs if d][:max_lines // 2]
    lines += [f for f in env.files if f.count(os.sep) == 0][:max_lines // 2]
    return "\n".join(lines)


def _readme(env):
    for f in env.files:
        if os.path.basename(f).lower().startswith("readme") and os.sep not in f:
            return scrub((env._text(f) or "")[:3000])
    return "(no README)"


def make_tasks(teacher, env, repo, n=4, rng=random):
    src = [f for f in env.files if re.search(r"\.(py|js|ts|tsx|rs|go|java|kt|rb|php|c|cc|cpp|h|hpp|cs|ex|scala|swift)$", f)
           and "test" not in f.lower()]
    if not src:
        return []
    path = rng.choice(src)
    code = env.read_file(path, 1, 200)
    msg = [{"role": "user", "content": TASK_PROMPT.format(
        repo=repo, readme=_readme(env), tree=_tree(env), path=path, code=code, n=n)}]
    out = teacher.chat(msg, max_tokens=1200, temperature=0.8, think=False)
    txt = out.get("content") or ""
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        tasks = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    good = []
    for t in tasks:
        if isinstance(t, dict) and isinstance(t.get("task"), str) and len(t["task"]) > 20:
            ap = t.get("answer_path")
            good.append({"task": t["task"], "kind": t.get("kind"),
                         "answer_path": ap if isinstance(ap, str) and ap in env.files else None})
    return good


def _args(call):
    a = call["function"].get("arguments") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            a = {}
    return a


def rollout(teacher, env, repo, task, max_turns=20):
    msgs = [{"role": "system", "content": SYSTEM.format(repo=repo)},
            {"role": "user", "content": task["task"]}]
    seen = {}
    for _ in range(max_turns):
        out = teacher.chat(msgs, tools=TOOLS)
        calls = out.get("tool_calls") or []
        turn = {"role": "assistant", "content": out.get("content") or ""}
        if out.get("reasoning_content"):
            turn["reasoning_content"] = out["reasoning_content"]
        if not calls:
            msgs.append(turn)
            return msgs, "answered"
        turn["tool_calls"] = [{"type": "function", "function": {
            "name": c["function"]["name"], "arguments": _args(c)}} for c in calls]
        msgs.append(turn)
        for c in turn["tool_calls"]:
            key = json.dumps(c["function"], sort_keys=True)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                return msgs, "loop"
            msgs.append({"role": "tool", "content": env.call(c["function"]["name"],
                                                              c["function"]["arguments"])})
    return msgs, "turn_budget"


def keep(msgs, status, task):
    if status != "answered":
        return False, status
    if not any(m.get("tool_calls") for m in msgs):
        return False, "no_tool_use"
    final = msgs[-1].get("content") or ""
    if len(final.strip()) < 20:
        return False, "empty_answer"
    if task.get("answer_path") and task["answer_path"] not in final:
        return False, "wrong_path"
    return True, "ok"


def build(repos, out_path, teacher, tasks_per_repo=4, seed=0, log=print):
    """repos: [(name, root_dir, meta)]. Appends kept episodes to out_path."""
    rng = random.Random(seed)
    stats = {}
    with open(out_path, "a") as fh:
        for name, root, meta in repos:
            env = RepoEnv(root)
            t0 = time.time()
            try:
                tasks = make_tasks(teacher, env, name, n=tasks_per_repo, rng=rng)
            except requests.RequestException as err:
                log(f"  {name}: task generation failed: {err}")
                continue
            for task in tasks:
                try:
                    msgs, status = rollout(teacher, env, name, task)
                except requests.RequestException as err:
                    status, msgs = f"http_error", []
                ok, why = keep(msgs, status, task) if msgs else (False, status)
                stats[why] = stats.get(why, 0) + 1
                if ok:
                    fh.write(json.dumps({"repo": name, "commit": meta.get("commit"),
                                         "license": meta.get("license"), "task": task,
                                         "tools": TOOLS, "messages": msgs}) + "\n")
                    fh.flush()
            log(f"  {name}: {len(tasks)} tasks in {time.time() - t0:.0f}s; totals {stats}")
    return stats
