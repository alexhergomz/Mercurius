"""Filter nvidia/Open-SWE-Traces into our episode format, under the data policy.

Each row carries the REPOSITORY's license, whether the trajectory resolved the
task, and which model produced it, so every policy filter applies per row
rather than per dataset (docs/data_policy.md):

  * repository license permissive (the dataset's CC-BY-4.0 covers the
    collection, not the code inside it);
  * generator's license permits training other models -- Qwen, DeepSeek, GLM,
    gpt-oss, Nemotron. Anything else, including models we cannot verify, is
    dropped rather than assumed;
  * resolved == 1 where the label exists, so trajectories that failed their own
    task are not taught as if they succeeded;
  * repository not one of the twelve SWE-bench Verified repositories, and not
    opted out of The Stack.

Streamed, never downloaded whole (the set is 42.6 GB), and stops at a token
budget. Output is the same jsonl shape the repository episodes use, so both
feed the one loader.

    python scripts/fetch_swe_traces.py --target-tokens 8000000
"""
import argparse
import json
import os
import re

from transformers import AutoTokenizer

from mercurius.data.repo_env import scrub
from mercurius.paths import DATA_DIR, ROOT, STAGE_AB
from scripts.repo_licenses import PERMISSIVE, SWE_BENCH_VERIFIED

# generator stated on the dataset card where rows do not carry it
SOURCE_GENERATOR = {"nemotron-swe-v1": "Qwen3-Coder-480B"}

GEN_OK = re.compile(r"^(qwen|deepseek|glm|gpt-oss|nemotron|llama-nemotron)", re.I)

# Task statements must be DERIVED FROM CODE, not copied from people. Verified
# from the source cards:
#   R2E-Gym   -- statements synthesised from the commit by their SWE-GEN
#                back-translation pipeline (arXiv:2504.07164), Apache-2.0
#   SWE-smith -- tasks synthesised by perturbing code, MIT
# Rejected, because their prompts ARE the human issue text and that text is not
# covered by the repository's licence (docs/data_policy.md):
#   Scale-SWE        -- card: "the issue description conveying the bug"; and it
#                       carries no licence tag at all
#   SWE-rebench-V2   -- card: "derived from real GitHub issues and pull requests"
#   SWE-Zero/SWE-Hero-- prompts are verbatim bug reports
STATEMENT_OK = re.compile(r"(r2e[-_]?gym|swe[-_]?smith)", re.I)
STATEMENT_BAD = re.compile(r"(scale[-_]?swe|swe[-_]?rebench|swe[-_]?bench[-_]?extra)", re.I)
# belt and braces: text that reads like a pasted human report even so
HUMAN_ISSUE = re.compile(
    r"(###\s*(Describe the bug|Steps to reproduce|Expected behaviou?r|To Reproduce)"
    r"|github\.com/[\w.-]+/[\w.-]+/(issues|pull)/\d+"
    r"|\B@[A-Za-z0-9-]{3,}\b.{0,40}(wrote|said|commented)"
    r"|<!--.*?(issue|bug).*?-->)", re.I | re.S)
# source -> (hf id, [(config, split)], how to read provenance / license)
SOURCES = {
    "open-swe-traces": ("nvidia/Open-SWE-Traces",
                        [("v1.2", "minisweagent"), ("v1.1", "minisweagent"),
                         ("v1.1", "sweagent"), ("v1.1", "openhands"),
                         ("v1.0", "sweagent"), ("v1.0", "openhands")],
                        "hf_dataset_name"),
    # R2E-Gym tasks: statements synthesised from the commit (SWE-GEN), per-row
    # repository licence, trajectories from Qwen3-Coder-480B; CC-BY-4.0.
    "nemotron-swe-v1": ("nvidia/Nemotron-SWE-v1", [("default", "r2e_gym")], "dataset"),
}


def convert(row):
    """Their message shape -> ours, and tools from JSON strings to objects."""
    msgs = []
    raw = row["messages"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    for m in raw:
        r = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("reasoning_content"):
            r["reasoning_content"] = m["reasoning_content"]
        tc = m.get("tool_calls")
        if tc:
            out = []
            for c in tc:
                fn = c.get("function", c)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                out.append({"type": "function",
                            "function": {"name": fn.get("name"), "arguments": args}})
            r["tool_calls"] = out
        msgs.append(r)
    tools = []
    for t in row.get("tools") or []:
        tools.append(json.loads(t) if isinstance(t, str) else t)
    return msgs, tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="nemotron-swe-v1", choices=sorted(SOURCES))
    ap.add_argument("--target-tokens", type=int, default=8_000_000)
    ap.add_argument("--max-tokens-per-row", type=int, default=65_536)
    ap.add_argument("--out", default=str(DATA_DIR / "episodes/swe_traces.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    # Rows sometimes give a LINK instead of an SPDX id ("...?tab=License-1-ov-file").
    # Resolve those against GitHub's own detection (scripts/repo_licenses.py's
    # cache, extended on demand) rather than dropping permissive repositories
    # over a formatting difference.
    lic_cache_path = ROOT / "data/repos/licenses.json"
    lic_cache = json.load(open(lic_cache_path)) if lic_cache_path.exists() else {}
    import subprocess
    import requests as _rq
    _tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
    _sess = _rq.Session()
    _sess.headers.update({"Authorization": f"Bearer {_tok}",
                          "Accept": "application/vnd.github+json"})

    def resolve_license(repo_name, stated):
        if stated in PERMISSIVE:
            return stated
        m = lic_cache.get(repo_name)
        if m and m.get("license"):
            return m["license"]
        try:
            r = _sess.get(f"https://api.github.com/repos/{repo_name}", timeout=20)
            spdx = ((r.json().get("license") or {}).get("spdx_id")
                    if r.status_code == 200 else None)
        except Exception:
            spdx = None
        lic_cache[repo_name] = {"license": spdx, "resolved_for_traces": True}
        return spdx

    optouts = json.load(open(ROOT / "data/repos/optouts.json"))
    opt_acc, opt_repo = set(optouts["accounts"]), set(optouts["repos"])
    verified = {x.lower() for x in SWE_BENCH_VERIFIED}
    seen = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            _r = json.loads(line)
            seen.add(_r.get("trajectory_id"))
    total, kept, drop = 0, 0, {}
    fh = open(a.out, "a")
    hf_id, configs, prov_key = SOURCES[a.source]
    for cfg, split in configs:
        if total >= a.target_tokens:
            break
        ds = load_dataset(hf_id, cfg, split=split, streaming=True)
        for row in ds:
            if total >= a.target_tokens:
                break
            tid = row.get("trajectory_id") or row.get("uuid") or row.get("instance_id")
            if tid in seen:
                continue
            repo = (row.get("repo") or "").lower()
            md = row.get("metadata") or {}
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except json.JSONDecodeError:
                    md = {}
            gen = ((md.get("teacher_model") or {}).get("name")
                   or SOURCE_GENERATOR.get(a.source, ""))
            prov = str(row.get(prov_key) or "")
            why = None
            lic = resolve_license(row.get("repo") or "", row.get("license"))
            if lic not in PERMISSIVE:
                why = f"license:{lic or row.get('license')}"
            elif not GEN_OK.match(gen or ""):
                why = f"generator:{gen}"
            elif row.get("resolved") not in (1, None, -1):
                why = "unresolved"
            elif repo in verified or repo.split("/")[0] in opt_acc or repo in opt_repo:
                why = "excluded_repo"
            elif STATEMENT_BAD.search(prov) or not STATEMENT_OK.search(prov):
                # unknown provenance is treated as human issue text, not assumed clean
                why = f"statement_source:{prov}"
            if why:
                drop[why] = drop.get(why, 0) + 1
                continue
            msgs, tools = convert(row)
            if any(HUMAN_ISSUE.search(m.get("content") or "") for m in msgs[:3]):
                drop["human_issue_text"] = drop.get("human_issue_text", 0) + 1
                continue
            for m in msgs:          # scrub every message, not only tool output
                if m.get("content"):
                    m["content"] = scrub(m["content"])
                if m.get("reasoning_content"):
                    m["reasoning_content"] = scrub(m["reasoning_content"])
            text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False)
            n = len(tok(text, add_special_tokens=False).input_ids)
            if n > a.max_tokens_per_row or n < 512:
                drop["length"] = drop.get("length", 0) + 1
                continue
            fh.write(json.dumps({"source": f"{a.source}/{cfg}/{split}",
                                 "dataset_id": hf_id, "provenance": prov,
                                 "trajectory_id": tid, "repo": row.get("repo"),
                                 "license": lic, "license_stated": row.get("license"),
                                 "language": row.get("language"),
                                 "generator": gen, "resolved": row.get("resolved"),
                                 "tools": tools, "messages": msgs, "n_tokens": n}) + "\n")
            fh.flush()
            kept += 1
            total += n
            if kept % 25 == 0:
                print(f"  {cfg}/{split}: kept {kept}, {total / 1e6:.2f} M tokens; drops {drop}",
                      flush=True)
    json.dump(lic_cache, open(lic_cache_path, "w"), indent=0)
    print(f"kept {kept} trajectories, {total / 1e6:.2f} M tokens -> {a.out}")
    print(f"drops: {drop}")


if __name__ == "__main__":
    main()
