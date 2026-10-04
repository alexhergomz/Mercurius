"""Full-set HumanEval / MBPP / GSM8K, batched, with the protocol details right.

The earlier runner was wrong in ways that all pushed scores DOWN:
  * 25 problems instead of 164 / 1319 / 257
  * a 320-token generation cap that truncated correct answers mid-sentence
    (worth +8 points on HumanEval and +13 on the held-out split when lifted)
  * HumanEval assembled as `extracted_code + test`, which scores a body-only
    completion as a failure instead of prepending the prompt and retrying
  * results written only at the very end, so a crash in the gsm8k scorer
    discarded 26 minutes of finished HumanEval

BATCHING is correct here, which was verified rather than assumed. Our model has
24 recurrent Gated-DeltaNet layers, so left-padding could in principle corrupt
the recurrence; measured:

    same prompt x3 (no padding difference)  -> diverges from batch-1 at token 18
    sorted, minimal padding                 -> identical, 40/40
    worst case (short + long)               -> identical, 40/40, INCLUDING the
                                               most-padded row

Padding is therefore handled (the GDN forward calls
`apply_mask_to_padding_states`, and a left-padded recurrence decays a zero state,
which stays zero). The one divergence is batched-GEMM reduction order flipping a
near-tie argmax -- deterministic per batch shape, present in any framework, and
the same reason batch-1 runs are not bit-reproducible across hardware.

Prompts are sorted by length before batching: less padding means less wasted
compute, and it keeps batches numerically closer to the batch-1 reference.

    python experiments/bench_full.py --arms masked=ckpt/adapters-masked150-best.pt \
        original=original --tasks humaneval gsm8k mbpp --batch 16
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter

import torch

from mercurius.paths import LOGS_DIR, ROOT, STAGE_AB

HE_PROMPT = ("Complete the following Python function. Reply with the complete "
             "function in a single ```python code block.\n\n```python\n{prompt}```\n")
MBPP_PROMPT = ("Write a Python function for this task. Reply with the function in a "
               "single ```python code block.\n\n{text}\n\nIt must satisfy:\n"
               "```python\n{test}\n```\n")
GSM_PROMPT = ("Solve the problem. Reason step by step, then give the final numeric "
              "answer on its own last line as `#### <number>`.\n\n{question}\n")


def strip_thinking(text):
    """Everything after `</think>`. Scoring the whole generation scored the
    model's SCRATCH WORK: 26 of 32 HumanEval generations had a code fence inside
    the think block and only 5 had one after it, so `extract_code` was returning
    draft code the model was still reasoning about.

    Split on the CLOSER, not a paired regex -- Qwen3 thinking models often emit
    `</think>` with no opening tag (stated on the model card).
    """
    return text.split("</think>")[-1] if "</think>" in text else text


DEFINES = re.compile(r"^\s*(?:def|class|from|import|async def)\s", re.M)
# a line that starts a block of code: def/class/import, a decorator, or a
# module-level assignment such as `NO_OF_CHARS = 256`
CODE_START = re.compile(
    r"^(?:from |import |def |class |async def |@\w|"
    r"[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*=(?!=))", re.M)


def extract_code(text):
    """The model's FINAL answer, not its first draft.

    This returned m[0] -- the first fence -- which scores whatever the model
    tried first. 44 of 257 MBPP generations emitted more than one fence and one
    emitted 74, because `enable_thinking=False` stops the `<think>` block but not
    the deliberation: the model reasons in prose and shows its drafts. On a
    controlled draft-then-correct input, first-fence scored 0/30.

    Taking the last fence UNCONDITIONALLY has its own failure: models often close
    with a usage example (```python\nprint(f(3))\n```) that defines nothing. So
    prefer the last fence that actually DEFINES something, and fall back to the
    last fence otherwise.

    Note what this deliberately does NOT do: try every fence and pass if any of
    them works. That is best-of-N over the model's own rejected drafts and would
    inflate the score. One fence is chosen, and its verdict stands.
    """
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        defining = [c for c in m if DEFINES.search(c)]
        return (defining or m)[-1]
    # No closed fence -- usually a truncated generation. Anchor at the first
    # CODE-LOOKING line and keep everything after it.
    #
    # Anchoring on `def|class|from|import` alone silently drops module-level
    # state that the function needs. MBPP/18 opens with `NO_OF_CHARS = 256` and
    # then defines four functions using it; starting at the first `def` produced
    # code that raises NameError, scoring a correct answer as wrong. Bare
    # assignments and decorators therefore anchor too.
    m = CODE_START.search(text)
    return text[m.start():] if m else text


def extract_number(text):
    for pat in (r"####\s*(-?\d[\d,]*(?:\.\d+)?)", r"(-?\d[\d,]*(?:\.\d+)?)"):
        m = re.findall(pat, text or "")
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                continue
    return None


def run_python(code, timeout=15):
    """Subprocess with a timeout. NOT a sandbox -- benchmark code only."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        p = fh.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, timeout=timeout,
                           text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(p)


IMPORT_RE = re.compile(r"^(?:from|import)\s+.*$", re.M)


def score_humaneval(out, row):
    out = strip_thinking(out)
    """Pass if ANY reasonable assembly of the completion runs the tests.

    Measured against the 40 canonical solutions: running the extracted code
    ALONE passes 40/40 when the model reproduces the prompt's import lines, and
    only 20/40 when it does not -- HumanEval prompts start with things like
    `from typing import List`, and a correct function that omits them dies on a
    NameError. Scoring one assembly therefore fails up to half the problems for
    a reason that has nothing to do with the model.

    So: code alone, then the prompt's imports + code, then the whole prompt +
    code. Any pass counts, which is what the completion-style protocol means.
    """
    code = extract_code(out)
    tail = "\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
    imports = "\n".join(IMPORT_RE.findall(row["prompt"]))
    for body in (code, imports + "\n" + code, row["prompt"] + code):
        if run_python(body + tail):
            return True
    return False


def score_mbpp(out, row):
    out = strip_thinking(out)
    code = extract_code(out)
    imports = "\n".join(row.get("test_imports") or [])
    tests = "\n".join(row["test_list"]) + "\n"
    for body in (f"{imports}\n{code}\n", f"{code}\n"):
        if run_python(body + tests):
            return True
    return False


def _sample(logits, temperature, top_p, top_k):
    """Qwen3/3.5 cards: "DO NOT use greedy decoding" -- it causes repetition and
    degradation. Thinking: T=0.6, top_p=0.95, top_k=20. Non-thinking: T=0.7,
    top_p=0.8, top_k=20.

    `temperature <= 0` still means argmax, because a diagnostic that compares two
    DECODE PATHS (cached vs recomputed) has to remove the sampler as a variable.
    But greedy is degenerate at this scale -- that is the documented reason the
    card forbids it -- so a benchmark must never reach it by accident, and
    `--temperature 0` is rejected at the CLI rather than here.

    A corollary that cost me a wrong inference: output degeneration observed
    UNDER GREEDY says nothing about whether a sampled run degenerates. Greedy
    repeating itself is the expected behaviour, not evidence.
    """
    if temperature <= 0:
        return logits.argmax(-1)
    lg = logits.float() / temperature
    if top_k:
        kth = lg.topk(min(top_k, lg.shape[-1]), dim=-1).values[:, -1:]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    probs = torch.softmax(lg, -1)
    if top_p and top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        cum = sp.cumsum(-1)
        sp[(cum - sp) > top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
    return torch.multinomial(probs, 1).squeeze(-1)


# The per-task generation budget. Diagnostics must IMPORT this rather than
# passing a literal: a small cap was hardcoded into three separate scripts in one
# day, each time silently censoring the very distribution being measured.
MAXNEW = {"humaneval": 32768, "gsm8k": 32768, "mbpp": 32768}
TRUNC_WARN = 0.02


@torch.no_grad()
def generate_batch(model, tok, texts, max_new, eos_ids,
                   temperature=0.6, top_p=0.95, top_k=20, _warned=[]):
    """Left-padded, attention mask extended each step, sampled not greedy.

    Warns once when the batch truncates above TRUNC_WARN. Every caller funnels
    through here, so this is the one place a censored measurement can be caught
    regardless of which script set the budget.
    """
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids, am = enc.input_ids.cuda(), enc.attention_mask.cuda()
    # logits_to_keep=1: compute the lm_head on the LAST position only.
    # Without it, prefill materialises logits for every position --
    # batch x prompt_len x 248,320 x 2 bytes, which is ~24 GB at batch 32 and
    # ~47 GB at batch 64 with 1500-token prompts. That, not batching itself, is
    # what exhausted memory: the KV cache here is MLA latents (~0.8 GB) and the
    # recurrent state ~3 GB, both negligible beside the logits tensor.
    out = model(ids, attention_mask=am, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    nxt = _sample(out.logits[:, -1], temperature, top_p, top_k)
    n = len(texts)
    toks = [[] for _ in range(n)]
    done = [False] * n
    for _ in range(max_new):
        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                if t in eos_ids:
                    done[i] = True
                else:
                    toks[i].append(t)
        if all(done):
            break
        am = torch.cat([am, torch.ones(n, 1, dtype=am.dtype, device=am.device)], 1)
        o = model(nxt.view(-1, 1), attention_mask=am, past_key_values=cache,
                  use_cache=True)
        cache = o.past_key_values
        nxt = _sample(o.logits[:, -1], temperature, top_p, top_k)
    # a row still not `done` ran out of budget rather than finishing: that is
    # truncation, and truncation silently turns correct answers into wrong ones
    # (worth +8 points on HumanEval and +13 on the held-out split today)
    # per-row truncation flags too: inferring them later by re-tokenising works
    # but is a reconstruction, and a run should record what it did.
    n_cut = sum(1 for d in done if not d)
    if n_cut / max(n, 1) > TRUNC_WARN and not _warned:
        _warned.append(1)
        print(f"  !! {n_cut}/{n} of this batch hit the {max_new}-token cap. Every "
              f"length statistic from this run is CENSORED, and truncated "
              f"generations score as failures. Raise max_new (see MAXNEW) or "
              f"shrink the item count instead.", flush=True)
    return ([tok.decode(t, skip_special_tokens=False) for t in toks],
            n_cut, [not d for d in done])


def load_rows(name, limit, seed=0):
    from datasets import load_dataset
    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    else:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)
    if not limit:
        return rows
    # A SEEDED RANDOM sample, not rows[:limit]. HumanEval is ordered roughly by
    # difficulty, so its first 25 problems scored 84% where the full set scores
    # ~49% -- the head of an ordered dataset is not a sample, and three numbers
    # reported earlier today were inflated by exactly that.
    import random
    r = random.Random(seed)
    idx = sorted(r.sample(range(len(rows)), min(limit, len(rows))))
    return [rows[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, metavar="NAME=CKPT")
    ap.add_argument("--dial", default="nope", choices=["nope", "k4", "k8", "c1", "k24", "c0"],
                    help="MUST match the arm's training --dial. build() applies "
                         "the rotary mask, which changes no tensor shapes, so a "
                         "mismatch silently evaluates a different model.")
    ap.add_argument("--head-swap", default=None, metavar="BLOB",
                    help="head-swap cache blob, required for any arm whose "
                         "checkpoint carries lm_head.proj.* . build() raises "
                         "rather than falling back to the student's own head, "
                         "because that would score a model that never existed.")
    ap.add_argument("--tasks", nargs="+", default=["humaneval", "gsm8k", "mbpp"],
                    choices=["humaneval", "mbpp", "gsm8k"])
    ap.add_argument("--limit", type=int, default=0, help="0 = FULL set")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=0,
                    help="0 = per-task defaults. A batch runs until its SLOWEST "
                         "member stops, so an over-generous cap costs every "
                         "sequence in the batch, not just the rambling one.")
    ap.add_argument("--covs", default=str(ROOT / "cache/kv_covs_4b_mix.pt"))
    ap.add_argument("--mla-groups", default=str(ROOT / "cache/mla_groups_retr_4096_mix.json"))
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-think", action="store_true",
                    help="pre-close the think block via the chat template. The "
                         "DEFAULT template opens `<think>` for the model, so it "
                         "is committed to reasoning: with a 512-token cap, 30 of "
                         "32 HumanEval generations never emitted `</think>` and "
                         "every score was budget-limited. Thinking-on measures "
                         "the ceiling; thinking-off measures what you would "
                         "deploy for simple completion. They are different "
                         "numbers and both are legitimate -- say which one you "
                         "are reporting.")
    ap.add_argument("--out", default=str(LOGS_DIR / "bench_full.json"))
    ap.add_argument("--qat", action="store_true",
                    help="rebuild as the DEPLOYED 4-bit model (models/qat.py) -- "
                         "required for checkpoints of a --qat run")
    ap.add_argument("--qat-kv-bits", type=int, default=4)
    ap.add_argument("--qat-kv-group", type=int, default=32)
    ap.add_argument("--qat-kv-rot", default="none", choices=["none", "orth"])
    ap.add_argument("--qat-kv-quant", default="int", choices=["int", "tq"],
                    help="KV latent quantizer: int = symmetric int, fp16 scale per "
                         "--qat-kv-group; tq = TurboQuant-MSE (no QJL): random rotation, "
                         "fp16 norm per token, Beta Lloyd-Max codebook (#66)")
    ap.add_argument("--qat-gate-bits", type=int, default=4)
    ap.add_argument("--qat-embed-bits", type=int, default=4)
    a = ap.parse_args()
    _qat = (dict(kv_bits=a.qat_kv_bits, kv_group=a.qat_kv_group, kv_rot=a.qat_kv_rot,
                 gate_bits=a.qat_gate_bits, embed_bits=a.qat_embed_bits,
                 kv_quant=a.qat_kv_quant)
            if a.qat else None)
    if a.temperature <= 0:
        # GREEDY IS THE LITERATURE CONVENTION (#50): lm-eval-harness gsm8k/humaneval/
        # mbpp set do_sample=False, Cobbe et al. and Qwen's own math harness use T=0,
        # and greedy pass@1 is what code papers report. The old refusal cited "the
        # Qwen3/3.5 cards" -- the Qwen3.5-4B card never mentions greedy (#50). The
        # real risk is repetition, so the truncation rate is reported per task and
        # must be read alongside the score.
        print("  decoding: GREEDY (argmax). Watch the reported truncation / repetition "
              "rate -- a degenerate arm shows up there first.", flush=True)
    from transformers import AutoTokenizer
    from mercurius.eval.retrieval_ab import build, build_original_nf4
    from mercurius import guard
    guard.cap_cuda_memory(60)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_ids = {tok.eos_token_id}
    for s in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0:
            eos_ids.add(i)
    report = {}
    if os.path.exists(a.out):
        report = json.load(open(a.out))
    for spec in a.arms:
        name, ckpt = spec.split("=", 1)
        print(f"\n=== {name} ({ckpt})", flush=True)
        model = (build_original_nf4() if ckpt in ("original", "base", "none")
                 else build(ckpt, a.dc, a.covs, quantize=True, groups=a.mla_groups,
                            head_swap=a.head_swap, dial=a.dial, qat=_qat))
        model.eval()
        report.setdefault(name, {})
        # Qwen3.5-4B's own card: 32,768 tokens for most queries, and the cap
        # covers THINKING + answer (confirmed for Anthropic, OpenAI o-series,
        # vLLM and HF generate alike). 512 was ~64x too small. See MAXNEW above.
        for task in a.tasks:
            max_new = a.max_new or MAXNEW[task]
            rows = load_rows(task, a.limit)
            fmt = (lambda r: HE_PROMPT.format(prompt=r["prompt"])) if task == "humaneval" \
                else (lambda r: MBPP_PROMPT.format(text=r["prompt"], test=r["test_list"][0])) \
                if task == "mbpp" else (lambda r: GSM_PROMPT.format(question=r["question"]))
            tk = {"enable_thinking": False} if a.no_think else {}
            texts = [tok.apply_chat_template([{"role": "user", "content": fmt(r)}],
                                             tokenize=False, add_generation_prompt=True,
                                             **tk)
                     for r in rows]
            # length-sorted so batches pad minimally
            order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
            st, t0, gens = Counter(), time.time(), []
            for b0 in range(0, len(order), a.batch):
                idxs = order[b0:b0 + a.batch]
                outs, n_trunc, was_cut = generate_batch(
                    model, tok, [texts[i] for i in idxs], max_new, eos_ids,
                    a.temperature, a.top_p, a.top_k)
                st["truncated"] += n_trunc
                for i, out, cut in zip(idxs, outs, was_cut):
                    r = rows[i]
                    gens.append({"task": task, "idx": i, "id": r.get("task_id", i),
                                 "out": out, "truncated": cut,
                                 "max_new": max_new})
                    if task == "gsm8k":
                        got = extract_number(strip_thinking(out))
                        want = extract_number(r["answer"])
                        ok = (got is not None and want is not None
                              and abs(got - want) < 1e-4)
                    elif task == "humaneval":
                        ok = score_humaneval(out, r)
                    else:
                        ok = score_mbpp(out, r)
                    st["pass" if ok else "fail"] += 1
                n = st["pass"] + st["fail"]
                el = (time.time() - t0) / 60
                print(f"  {task} {n}/{len(rows)}  pass {st['pass']} ({st['pass']/n:.1%})"
                      f"  trunc {st['truncated']}  {el:.1f} min", flush=True)
                report[name][task] = {"pass": st["pass"], "n": n,
                                      "truncated": st["truncated"],
                                      "rate": st["pass"] / max(n, 1),
                                      "full_set": not a.limit, "batch": a.batch,
                                      "max_new": max_new}
                json.dump(report, open(a.out, "w"), indent=1)   # checkpoint
            # generations on disk: a scoring bug then costs a re-score, not a rerun
            gp = a.out.replace(".json", f"_{name}_{task}"
                           f"{'_nothink' if a.no_think else ''}_gens.jsonl")
            with open(gp, "w") as fh:
                for g in gens:
                    fh.write(json.dumps(g) + "\n")
            print(f"  {name} {task}: {st['pass']}/{st['pass']+st['fail']} = "
                  f"{st['pass']/max(st['pass']+st['fail'],1):.1%}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
