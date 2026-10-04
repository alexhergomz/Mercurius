"""Score an arm on the held-out task split, through the real tool loop.

The student generates in-process: render the chat template with the tool schemas,
greedy-decode an assistant turn, parse any <tool_call>, execute it against the
real RepoEnv, append the result, repeat. Same environment, same tools, same
answer format as training -- so the number means "can this model do the task",
not "does it like the right tokens".

Scored by the parser oracle (code_graph.verify), the same one that admitted the
training data, so scoring is mechanical and there is nothing to trust.

    python experiments/score_capability.py --arms pilot=ckpt/adapters-pilotmix-best.pt \
        --covs cache/kv_covs_4b_mix.pt --mla-groups cache/mla_groups_retr_4096_mix.json
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import torch

from mercurius.data.code_graph import verify
from mercurius.data.repo_env import TOOLS, RepoEnv
from mercurius.paths import DATA_DIR, LOGS_DIR, ROOT, STAGE_AB

HELD_OUT = DATA_DIR / "eval" / "held_out_tasks.json"
SUFFIX = ("\n\nWork it out using the tools; do not guess. When you are certain, "
          "give the answer in the <answer> tags exactly as specified, as the "
          "last thing in your reply.")
# Qwen3.5 emits tool calls as XML, not JSON:
#   <tool_call><function=grep><parameter=pattern>foo</parameter></function></tool_call>
# Parsing it as JSON silently yielded zero calls, so every task ended after one
# turn with no answer tag -- a uniform 0% that looked like the model could not
# do the task at all. The JSON form is kept as a fallback since the API returns
# that shape and some templates render it.
CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FN = re.compile(r"<function=([\w.]+)\s*>(.*?)</function>", re.S)
_PARAM = re.compile(r"<parameter=([\w.]+)\s*>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text):
    """[(name, args_dict)] from either the XML or the JSON form."""
    out = []
    for block in CALL_RE.findall(text):
        for name, body in _FN.findall(block):
            args = {}
            for k, v in _PARAM.findall(body):
                v = v.strip()
                if re.fullmatch(r"-?\d+", v):
                    args[k] = int(v)
                else:
                    args[k] = v
            out.append((name, args))
        if not _FN.search(block):
            try:
                j = json.loads(block.strip())
                out.append((j.get("name"), j.get("arguments") or {}))
            except json.JSONDecodeError:
                pass
    return out


@torch.no_grad()
def gen_turn(model, tok, text, max_new=700):
    """Greedy continuation. Chunked prefill is not needed at these lengths, but
    the lm_head is applied only to the last position, never the whole sequence."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    out = model(ids, use_cache=True)
    cache, nxt = out.past_key_values, out.logits[0, -1].argmax()
    toks = []
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    for _ in range(max_new):
        if int(nxt) in eos:
            break
        toks.append(int(nxt))
        o = model(nxt.view(1, 1), past_key_values=cache, use_cache=True)
        cache, nxt = o.past_key_values, o.logits[0, -1].argmax()
    return tok.decode(toks, skip_special_tokens=False)


def run_task(model, tok, task, env, max_turns=12, max_new=700):
    """max_new matters more than it looks. At 320 the model was solving tasks and
    being cut off mid-sentence before it could emit <answer>: one trace found
    both gold files AND correctly excluded the header as a definition, then
    truncated. 9 of 10 failures scored `no_answer_tag` were substantially
    truncation, not inability."""
    msgs = [{"role": "user", "content": task["question"] + SUFFIX}]
    for _ in range(max_turns):
        text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                       add_generation_prompt=True)
        reply = gen_turn(model, tok, text, max_new=max_new)
        calls = parse_calls(reply)
        msgs.append({"role": "assistant", "content": reply})
        if not calls:
            return msgs, verify(task, reply)
        for name, args in calls[:2]:
            try:
                res = env.call(name, args)
            except Exception as err:
                res = f"Error: {type(err).__name__}: {err}"
            msgs.append({"role": "tool", "content": str(res)[:4000]})
    return msgs, (False, "turn_budget")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, metavar="NAME=CKPT")
    ap.add_argument("--covs", default=str(ROOT / "cache/kv_covs_4b_mix.pt"))
    ap.add_argument("--mla-groups", default=str(ROOT / "cache/mla_groups_retr_4096_mix.json"))
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = all held-out tasks")
    ap.add_argument("--quantize", action="store_true", default=True)
    ap.add_argument("--out", default=str(LOGS_DIR / "capability.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from mercurius.eval.retrieval_ab import build
    from mercurius import guard
    guard.cap_cuda_memory(60)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    tasks = json.load(open(HELD_OUT))["tasks"]
    if a.limit:
        tasks = tasks[:a.limit]
    clones = json.load(open(ROOT / "data/repos/clones.json"))
    envs = {}
    report = {}
    for spec in a.arms:
        name, ckpt = spec.split("=", 1)
        print(f"\n=== {name} ({ckpt})", flush=True)
        # build(adapters, dc, covs_path, ..., quantize=, groups=)
        if ckpt in ("original", "base", "none"):
            from mercurius.eval.retrieval_ab import build_original_nf4
            model = build_original_nf4()
        else:
            model = build(ckpt, a.dc, a.covs, quantize=a.quantize, groups=a.mla_groups)
        model.eval()
        st, t0 = Counter(), time.time()
        for i, t in enumerate(tasks, 1):
            repo = t["repo"]
            if repo not in envs:
                envs[repo] = RepoEnv(os.path.join(ROOT, clones[repo]["path"]),
                                     max_read_lines=400, search_k=5)
            try:
                _msgs, (ok, why) = run_task(model, tok, t, envs[repo])
            except Exception as err:
                ok, why = False, f"exc:{type(err).__name__}"
            st[f"{'pass' if ok else 'fail'}:{t['kind']}"] += 1
            st["pass" if ok else "fail"] += 1
            if not ok:
                st[f"why:{why.split(':')[0]}"] += 1
            if i % 10 == 0:
                print(f"  {i}/{len(tasks)}  pass {st['pass']}/{i} "
                      f"({st['pass'] / i:.0%})  {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        n = st["pass"] + st["fail"]
        report[name] = {"pass": st["pass"], "n": n,
                        "rate": st["pass"] / max(n, 1), **st}
        print(f"  {name}: {st['pass']}/{n} = {st['pass'] / max(n, 1):.1%} "
              f"in {(time.time() - t0) / 60:.1f} min")
        for k in sorted(st):
            if k.startswith(("pass:", "fail:", "why:")):
                print(f"      {k:<22} {st[k]}")
        del model
        torch.cuda.empty_cache()
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
