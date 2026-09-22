"""RULER-style NIAH evaluation of the converted model against the original.

Two metrics per sample, deliberately:

  EM   RULER's own string_match_all on greedy continuation -- the fraction of
       gold strings that appear in the generated text. Comparable in KIND to
       published RULER numbers. Its floor is 0, and a 0.8B model can sit on
       that floor, at which point it separates nothing.

  NLL  mean negative log-likelihood of the gold answer span, teacher-forced
       after the same prompt. Graded, no floor, one forward. This is what
       resolves arms that EM cannot.

Both are needed. EM alone can report 0.00 for every arm and hide a real
difference; NLL alone is not comparable to anything published.

The original is ONE ARM, not a gate. An earlier version of this plan screened
tasks by whether the original scored above floor, which silently assumes the
original bounds the converted model. It does not: the converted model is 12.7%
BETTER in perplexity (findings 0.3). A task where the original floors can still
be one where a converted arm does not, and screening on the original would
discard exactly that result. Screen on the MAX over arms, or not at all.

Every arm sees byte-identical samples: they are generated once, before any model
is built, and reused.
"""
import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mercurius.eval.retrieval_ab import build, build_original, build_original_nf4
from mercurius import guard
from mercurius.recovery.train import CKPT
from mercurius.eval import ruler_gen as R
from mercurius.paths import CACHE_DIR, CKPT_DIR

DEFAULT_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
                 "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                 "niah_multivalue", "niah_multiquery"]


def gold_string(outputs):
    """How the answer_prefix would naturally continue."""
    if len(outputs) == 1:
        return " " + outputs[0]
    return " " + ", ".join(outputs[:-1]) + ", and " + outputs[-1]


def gold_value_mask(outputs, tok):
    """Tokenize the gold continuation ONCE, and mark which tokens are values.

    Averaging the likelihood over the whole span dilutes the only tokens that
    require retrieval. Across three arms whose exact match spans 9.8 points, the
    full-span NLL spanned 0.001 nats -- effectively blind. Same averaging failure
    LongPPL identifies for long-context perplexity (arXiv:2410.23771): likelihood
    on ANSWER tokens tracks long-context accuracy, likelihood on the surrounding
    tokens does not. Here the surroundings are ", " and ", and ", which every arm
    predicts perfectly and which therefore pull every arm to the same number.

    The mask is built from CHARACTER OFFSETS, not by tokenizing each piece
    separately. Piecewise tokenization changes the sequence: a leading space
    merges with the first character of a value, so " e4a1..." is [378, 19, ...]
    whole but [220, 68, 19, ...] in pieces. Scoring pieces would measure a
    sequence the model never sees -- verified on numbers and uuids alike.
    """
    gold = gold_string(outputs)
    enc = tok(gold, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    spans, at = [], 0
    for o in outputs:
        i = gold.index(o, at)
        spans.append((i, i + len(o)))
        at = i + len(o)
    mask = [any(a < e and b > s for s, e in spans)
            for (a, b) in enc["offset_mapping"]]
    return ids, mask


@torch.no_grad()
def score_sample(model, tok, s, gen_tokens):
    """(span NLL, value-token NLL, greedy prediction) from ONE prefill.

    The prompt is the whole cost at long context (a 128k prefill is minutes;
    the answer is tens of tokens), and the previous version paid it twice: a
    teacher-forced pass for the NLL, then generate() from scratch for EM. Here
    the prompt is prefilled once with a cache; the gold span is scored on a
    COPY of that cache, and greedy decoding runs on the original.
    """
    import copy
    prompt_ids = tok(s["input"] + s["answer_prefix"]).input_ids
    gold_ids, val_mask = gold_value_mask(s["outputs"], tok)
    if gen_tokens < 0:
        # Budget the generation to the ANSWER, per sample. A flat cap silently
        # truncates the long answers: a uuid is ~25 tokens and four numbers with
        # separators is ~40, so a 24-token cap reported EM 0.00% on
        # niah_single_3 while the teacher-forced NLL was 0.064. Slack of 16
        # covers a short preamble before the answer.
        #
        # Floored at RULER's own 128. gold+16 alone capped niah_multiquery EM
        # at ~75% for EVERY arm, originals included: the gold string is the
        # four values, but models answer "- key: value" per line, so the keys
        # consumed the budget and the 4th value was cut mid-number.
        gen_tokens = max(128, len(gold_ids) + 16)
    pi = torch.tensor([prompt_ids], device="cuda")
    out = model(input_ids=pi, use_cache=True, logits_to_keep=1)
    first = out.logits[0, -1].float()                 # predicts gold[0]
    cache = out.past_key_values
    del out

    # NLL: gold[0] from the prompt's last position, then gold[1:] ONE TOKEN AT
    # A TIME on a copy of the cache. Not one multi-token forward: the hybrid
    # model only reads its linear-attention cache when seq_len == 1 (stock
    # Qwen3.5 and our port alike), so a multi-token continuation silently
    # restarts the recurrent state from zero -- measured NLL 2.7 against a
    # teacher-forced 0.06. The span is ~10-40 tokens, negligible against the
    # prefill. Token ids are concatenated, never strings.
    g = torch.tensor([gold_ids], device="cuda")
    logits = [first.unsqueeze(0)]
    c2 = copy.deepcopy(cache) if len(gold_ids) > 1 else None
    for t in range(len(gold_ids) - 1):
        o2 = model(input_ids=g[:, t:t + 1], past_key_values=c2, use_cache=True)
        c2 = o2.past_key_values
        logits.append(o2.logits[0, -1:].float())
        del o2
    del c2
    sl = torch.cat(logits, 0)
    assert sl.shape[0] == len(gold_ids), (sl.shape, len(gold_ids))
    per_tok = F.cross_entropy(sl, g[0], reduction="none")
    nll = per_tok.mean().item()
    m = torch.tensor(val_mask, device=per_tok.device)
    nll_v = per_tok[m].mean().item() if bool(m.any()) else nll

    pred = ""
    if gen_tokens:
        # greedy decode on the original cache (experiments/test_generate_cache
        # checks cached decoding against cache-free decoding, token for token)
        nxt, toks = first.argmax(), []
        for _ in range(gen_tokens):
            t = int(nxt)
            if t == tok.eos_token_id:
                break
            toks.append(t)
            o = model(input_ids=nxt.view(1, 1), past_key_values=cache,
                      use_cache=True)
            cache = o.past_key_values
            nxt = o.logits[0, -1].argmax()
        pred = tok.decode(toks, skip_special_tokens=True)
    del cache
    torch.cuda.empty_cache()
    return nll, nll_v, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="tag=path entries")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--em-samples", type=int, default=0, metavar="M",
                    help="score EM on only the first M samples of each cell "
                         "(0 = all). EM needs a second pass over the prompt plus "
                         "decode steps, so it is roughly half the runtime, and it "
                         "is saturated at 100%% in most cells while NLL is not.")
    ap.add_argument("--gen-tokens", type=int, default=-1,
                    help="-1 budgets per sample from the gold length "
                         "(recommended); 0 disables EM scoring; >0 is a flat cap")
    ap.add_argument("--init-adapters", default=None,
                    help="replayed before the checkpoint for runs that started "
                         "from earlier adapters (the 0.8B arms used "
                         "ckpt/adapters-combined.pt); none for a fresh run")
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--quantize", action="store_true",
                    help="rebuild converted arms with the trainer's NF4 step "
                         "(--student-bits 4); ORIGINAL_NF4 is the matching baseline")
    ap.add_argument("--mem-cap-gb", type=float, default=90.0)
    ap.add_argument("--gpu-temp-pause", type=float, default=84.0)
    ap.add_argument("--gpu-temp-resume", type=float, default=80.0)
    ap.add_argument("--alloc", default=None,
                    help="explicit per-layer d_c as a JSON dict or a path to one; "
                         "a retrieval_heads.json is accepted and the key named by "
                         "--alloc-key is used. Overrides --dc.")
    ap.add_argument("--alloc-key", default="retrieval",
                    help="which allocation inside a retrieval_heads.json to use: "
                         "uniform, spectral or retrieval")
    ap.add_argument("--covs", default=str(CACHE_DIR / 'kv_covs_4b.pt'))
    ap.add_argument("--out", default="logs/ruler.json")
    a = ap.parse_args()
    guard.cap_cuda_memory(a.mem_cap_gb)
    pacer = guard.ThermalPacer(a.gpu_temp_pause, a.gpu_temp_resume, 90.0, 85.0)

    _alloc = None
    if a.alloc:
        import os as _o
        _alloc = json.load(open(a.alloc)) if _o.exists(a.alloc) else json.loads(a.alloc)
        if a.alloc_key in _alloc:
            _alloc = _alloc[a.alloc_key]
        _alloc = {int(k): int(v) for k, v in _alloc.items()}
        print(f"  d_c allocation ({a.alloc_key}), total {sum(_alloc.values())}: "
              f"{dict(sorted(_alloc.items()))}", flush=True)

    tok = AutoTokenizer.from_pretrained(CKPT)

    print("generating samples (shared by every arm)", flush=True)
    data = {}
    for task in a.tasks:
        for n in a.lengths:
            data[(task, n)] = R.generate(task, n, a.samples, tok, verbose=True)
    print(f"  {sum(len(v) for v in data.values())} samples\n", flush=True)

    rows = []
    for spec in a.arms:
        tag, path = spec.split("=", 1)
        t0 = time.perf_counter()
        m = (build_original() if path == "ORIGINAL"
             else build_original_nf4() if path == "ORIGINAL_NF4"
             else build(path, a.dc, a.covs, init_adapters=a.init_adapters,
                        alloc=_alloc, quantize=a.quantize))
        pacer.attach(m)
        res = {}
        for task in a.tasks:
            for n in a.lengths:
                nlls, nllv, preds, refs, oom = [], [], [], [], 0
                lim = a.em_samples or len(data[(task, n)])
                for si, s in enumerate(data[(task, n)]):
                    # One OOM at the longest length must not destroy the whole
                    # sweep. Record the gap and carry on.
                    try:
                        nll, nll_v, pred = score_sample(
                            m, tok, s, a.gen_tokens if si < lim else 0)
                    except torch.cuda.OutOfMemoryError:
                        oom += 1
                        torch.cuda.empty_cache()
                        continue
                    nlls.append(nll); nllv.append(nll_v)
                    if si < lim:
                        preds.append(pred); refs.append(s["outputs"])
                if not nlls:
                    res[f"{task}@{n}"] = {"nll": None, "em": None, "oom": oom}
                    print(f"  {tag:<14} {task:<16}@{n:<6} "
                          f"all {oom} samples OOMed", flush=True)
                    continue
                em = (R.string_match_all(preds, refs)
                      if a.gen_tokens and preds else None)
                res[f"{task}@{n}"] = {
                    "nll": sum(nlls) / len(nlls),
                    "nll_v": sum(nllv) / len(nllv),
                    "em": em,
                    "n": len(nlls),
                    "n_em": len(preds),
                    "oom": oom,
                    "example_pred": preds[0][:80] if preds else "",
                }
                print(f"  {tag:<14} {task:<16}@{n:<6} "
                      f"NLL {res[f'{task}@{n}']['nll']:6.3f}"
                      + (f"   EM {em:6.2f}%" if em is not None else "")
                      + (f"   ({oom} OOM)" if oom else ""),
                      flush=True)
                # Per CELL, not per arm. Checkpointing only when an arm finished
                # meant an interruption 20 cells into a 24-cell arm lost all 20.
                json.dump(rows + [{"arm": tag, "res": res, "partial": True}],
                          open(a.out, "w"), indent=1)
        rows.append({"arm": tag, "res": res})
        # Dump after EVERY arm. A sweep this long must not be all-or-nothing.
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"  ({time.perf_counter() - t0:.0f}s, partial results in "
              f"{a.out.split('/')[-1]})\n", flush=True)
        pacer.detach()
        del m
        torch.cuda.empty_cache()

    keys = [f"{t}@{n}" for t in a.tasks for n in a.lengths]
    print("\nNLL of the gold span (lower is better)")
    hdr = f"{'task':<24}" + "".join(f"{r['arm'][:11]:>12}" for r in rows)
    print(hdr); print("-" * len(hdr))
    for k in keys:
        print(f"{k:<24}" + "".join(
            f"{r['res'][k]['nll']:>12.3f}" if r["res"].get(k, {}).get("nll")
            is not None else f"{'--':>12}" for r in rows))
    if a.gen_tokens:
        print("\nEM recall, RULER string_match_all (higher is better)")
        print(hdr); print("-" * len(hdr))
        for k in keys:
            print(f"{k:<24}" + "".join(
                f"{r['res'][k]['em']:>11.2f}%" if r["res"].get(k, {}).get("em")
                is not None else f"{'--':>12}" for r in rows))
        mx = {k: max((r["res"].get(k, {}).get("em") or 0.0) for r in rows)
              for k in keys}
        dead = [k for k in keys if mx[k] == 0.0]
        if dead:
            print(f"\n  EM is 0 for EVERY arm on {len(dead)}/{len(keys)} settings "
                  f"-- those separate nothing at this model size; read NLL there.")
            print(f"  {', '.join(dead)}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
