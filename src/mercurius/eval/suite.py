"""Richer evaluation: CE, top-k accuracy, teacher agreement, and generation.

Perplexity alone hides two things:

  * whether TOP-K DECODING degraded. CE is exactly log(PPL) so it adds no
    independent information -- it is reported because small deltas read more
    clearly in nats than in exponentiated form. What genuinely answers the
    decoding question is top-k accuracy and agreement with the teacher's top-k,
    since a model can hold its average CE while its argmax drifts.

  * whether GENERATION degraded. A model can improve perplexity while producing
    worse text -- distillation on one domain especially. Samples are the only
    way to see that.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def ce_and_topk(model, ids, n, ks=(1, 5, 10), chunk=1024, teacher=None,
                before_forward=None):
    """Cross-entropy, top-k accuracy vs ground truth, and top-1/k agreement
    with a teacher if supplied. Chunked: vocab is 248,320."""
    x = ids[:n].unsqueeze(0).cuda()
    # Ask for hidden states and suppress the full logit tensor. The lm_head is
    # applied chunk-wise below instead, because at 8192 x 248,320 the full
    # logits are 4.1 GiB resident for the whole eval -- and two concurrent evals
    # in that state hard-reset this machine on 2026-09-20. Chunked, the peak is
    # one chunk's worth and it is freed each iteration.
    # Final hidden states straight from the trunk (post final-norm, the tensor
    # lm_head consumes -- the trainer reads the same thing), with the lm_head
    # applied chunk-wise below. The previous version asked for
    # output_hidden_states=True, which keeps EVERY layer's states: 65 x 8192 x
    # 5120 for a 27B teacher, ~5 GiB, at the moment unified memory was already
    # tightest. That eval is where the 2026-09-21 machine reset happened.
    from mercurius.surgery.norm_fusion import get_trunk
    if before_forward is not None:      # e.g. a thermal cool-down
        before_forward()
    h = get_trunk(model)(input_ids=x).last_hidden_state[0]
    # After a head swap the output path is logits = W_t(P h): there is no single
    # weight matrix, and W_t P would be 635.7M to materialise for an eval. Project
    # the hidden state into the teacher's basis instead and use its head. The
    # ProjectedHead raises on `.weight` rather than returning something
    # plausible, so this is a loud failure to fix, not a silent wrong number.
    _head = model.get_output_embeddings()
    if hasattr(_head, "project"):
        h = _head.project(h)
        W_lm = _head.head.weight
    else:
        W_lm = _head.weight
    tgt = x[0, 1:]

    t_h = None
    if teacher is not None:
        if before_forward is not None:
            before_forward()
        t_h = get_trunk(teacher)(input_ids=x).last_hidden_state[0]
        _th = teacher.get_output_embeddings()
        if hasattr(_th, "project"):
            t_h = _th.project(t_h)
            t_W = _th.head.weight
        else:
            t_W = _th.weight

    tot_ce, cnt = 0.0, 0
    hits = {k: 0 for k in ks}
    agree = {k: 0 for k in ks}
    for i in range(0, n - 1, chunk):
        j = min(i + chunk, n - 1)
        sl = (h[i:j] @ W_lm.T).float()
        tg = tgt[i:j]
        tot_ce += F.cross_entropy(sl, tg, reduction="sum").item()
        cnt += j - i
        top = sl.topk(max(ks), dim=-1).indices
        for k in ks:
            hits[k] += (top[:, :k] == tg.unsqueeze(-1)).any(-1).sum().item()
        if t_h is not None:
            tt = (t_h[i:j] @ t_W.T).float().topk(max(ks), dim=-1).indices
            for k in ks:
                # does the teacher's argmax appear in the student's top-k?
                agree[k] += (top[:, :k] == tt[:, :1]).any(-1).sum().item()
        del sl
    del h
    if t_h is not None:
        del t_h
    torch.cuda.empty_cache()

    ce = tot_ce / cnt
    res = {"ce": ce, "ppl": float(torch.tensor(ce).exp()),
           **{f"top{k}": hits[k] / cnt * 100 for k in ks}}
    if teacher is not None:
        res.update({f"agree_top{k}": agree[k] / cnt * 100 for k in ks})
    return res


@torch.no_grad()
def sample(model, tok, prompts, max_new=96, temperature=0.7, top_p=0.9, seed=0):
    """Greedy-ish sampling. Kept simple and dependency-free so it works on the
    custom KDA model without relying on generate()'s cache plumbing."""
    torch.manual_seed(seed)
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        cur = ids
        for _ in range(max_new):
            logits = model(input_ids=cur).logits[0, -1].float()
            if temperature <= 0:
                nxt = logits.argmax()
            else:
                probs = F.softmax(logits / temperature, -1)
                sp, si = probs.sort(descending=True)
                keep = (sp.cumsum(-1) - sp) < top_p
                sp = sp * keep
                sp = sp / sp.sum()
                nxt = si[torch.multinomial(sp, 1)]
            cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
            if nxt.item() == tok.eos_token_id:
                break
        outs.append(tok.decode(cur[0, ids.shape[1]:], skip_special_tokens=True))
        torch.cuda.empty_cache()
    return outs


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h?\nA: Let's think step by step.",
    "The Jetson AGX Orin is an embedded computing platform that",
]


def report(tag, res):
    line = (f"  {tag:<26} CE {res['ce']:6.4f}  ppl {res['ppl']:7.3f}  "
            f"top1 {res['top1']:5.2f}%  top5 {res['top5']:5.2f}%  "
            f"top10 {res['top10']:5.2f}%")
    if "agree_top1" in res:
        line += (f"  | teacher-agree top1 {res['agree_top1']:5.2f}%"
                 f" top5 {res['agree_top5']:5.2f}%")
    print(line, flush=True)


# --- termination / stuck-rate -------------------------------------------------
# Roadmap fact 4 and D16: the failure this pipeline is trying to fix is that the
# model does not TERMINATE -- it restates itself until the budget runs out. That
# is invisible to every teacher-forced metric, because perplexity never asks the
# model to generate. Measured 2026-09-24 on the benchmark suite: the base NF4
# model had 0 stuck generations, ours had 28 on MBPP (10.9% of all). Selecting
# checkpoints on ppl is therefore selecting on a metric that cannot see the
# defect -- the same trap SEAL's offline-selection result describes, where
# validation loss was uncorrelated with closed-loop success and picked the worst
# model in the pool.
#
# Three detectors were tried on the same generations and gave 1.4%, 71% and
# 12.4%. Exact token periodicity misses a model restating itself in new words;
# duplicate-line fraction is ~collinear with "hit the cap" and scores a genuine
# redraft the same as a copy. What separates them is whether the model is still
# making PROGRESS, so novelty in the final quarter is what is measured here, and
# it validates: generations that terminated had a median novel-tail of 1.00.

def _novel_tail(text, min_len=20, quarter=0.75):
    """DEPRECATED -- kept only to re-read old dumps. Use _tail_entropy.

    Exact-line novelty over lines >20 chars, needing 12 of them. It fails on the
    cases it exists to catch, measured 2026-09-25 over 144 generations:

      * returns None for 23 of 48 SWAP generations INCLUDING TWO AT THE 6144
        CAP. The worst output is one unbroken line of "1122222222..." with no
        newlines, so `len(lines) < 12` and maximal degeneracy scores as
        unmeasurable.
      * returns 1.000 on "255, 256, 257, ..." -- every line is textually
        distinct, so patterned repetition is invisible to exact matching. Two
        such per arm sat among the ten longest generations and were NOT flagged.
      * callers divide n_stuck by len(prompts) while only the scoreable rows can
        contribute, so every rate it produced is biased DOWN. Re-scored, HIDDEN
        is 27.1% degenerate where this reported 14.6%.
    """
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > min_len]
    if len(lines) < 12:
        return None
    k = int(len(lines) * quarter)
    seen = set(lines[:k])
    tail = lines[k:]
    return sum(1 for l in tail if l not in seen) / len(tail)


def _tail_entropy(text, quarter=0.75, min_chars=200):
    """Degeneracy as COMPRESSIBILITY of the tail, on characters.

    zlib on the last quarter, ratio = compressed/raw. Robust exactly where
    line-matching is not: it needs no newlines, and it catches patterned
    repetition (incrementing counters, rotating templates) that is distinct
    line by line but carries almost no information.

    Measured over 144 generations from three arms: natural code and prose land
    at 0.25-0.71 (medians 0.49-0.58), degenerate loops at 0.010-0.036. The gap
    is two decimal orders with nothing in between, which is why a single
    threshold works and why DEGENERATE_BELOW sits at 0.12 rather than being
    tuned. Returns None only for genuinely short text, where a compression
    ratio is dominated by zlib's header.
    """
    import zlib
    t = text[int(len(text) * quarter):]
    if len(t) < min_chars:
        return None
    return len(zlib.compress(t.encode("utf-8", "replace"), 6)) / len(t)


DEGENERATE_BELOW = 0.12


@torch.no_grad()
def stuck_rate(model, tok, prompts, max_new=2048, temperature=0.7, top_p=0.8,
               top_k=20, stuck_below=0.05, seed=0, batch=16, dump=None):
    """Fraction of generations that stop producing anything new.

    Sampled, never greedy: the Qwen3/3.5 cards forbid greedy decoding because it
    degenerates, so a greedy run would measure the sampler rather than the model.

    BUDGET. max_new must be large enough that terminating generations actually
    terminate, or `hit cap` measures the budget rather than the model and
    `novel_tail` is computed on truncated text. At 512 this saturated: 22 of 24
    generations hit the cap, and the benchmark had already shown 28% truncation
    at 768 on the same class of prompt.

    SEEDED. Two consecutive evals of the SAME checkpoint on the SAME prompts
    returned stuck 17%/0% and novel-tail 0.70/0.88 -- +-0.18 from sampling alone,
    wider than any trend it was being used to read.

    BATCHED. An earlier version generated one prompt at a time, which is batch-1
    decode at ~12 tok/s: 48 prompts x 2048 tokens took over an hour per model.
    generate_batch is the same path the benchmark uses (164 HumanEval prompts in
    9.6 min at batch 64) and is verified correct under padding.
    """
    from experiments.bench_full import generate_batch
    torch.manual_seed(seed)
    was_training = model.training
    model.eval()
    side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            eos.add(i)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # pad less
    outs = [None] * len(texts)
    truncated = [None] * len(texts)
    n_trunc = 0
    for b0 in range(0, len(order), batch):
        idxs = order[b0:b0 + batch]
        gen, nt, cut = generate_batch(model, tok, [texts[i] for i in idxs],
                                      max_new, eos, temperature, top_p, top_k)
        n_trunc += nt
        for i, g, c in zip(idxs, gen, cut):
            outs[i] = g
            truncated[i] = c
    tok.padding_side = side
    if was_training:
        model.train()
    if dump:
        # A rate is not a diagnosis. Three detectors gave 1.4%, 71% and 12.4% on
        # the same generations and only reading them settled which was measuring
        # anything, so the text is written out alongside the score.
        import json as _json
        with open(dump, "w") as fh:
            for pr, o, c in zip(prompts, outs, truncated):
                nt = _novel_tail(o)
                # `truncated` is AUTHORITATIVE and `n_tok` is NOT. n_tok
                # re-tokenises the decoded text, which drifts from the real
                # generated length -- measured 2026-09-25, four capped HIDDEN
                # rows came back as 6138/6142/6143 against a 6144 cap, so a
                # "stopped before the cap" test counted them as TERMINATIONS
                # and put a spurious nonzero in the top hazard bin. Any
                # termination analysis must read `truncated`, never compare
                # n_tok to max_new.
                fh.write(_json.dumps({"prompt": pr, "gen": o,
                                      "novel_tail": nt, "truncated": c,
                                      "max_new": max_new,
                                      "n_tok": len(tok(o, add_special_tokens=False).input_ids)}) + "\n")
    tails = [_novel_tail(o) for o in outs]
    tails = [t for t in tails if t is not None]
    n_stuck = sum(1 for t in tails if t < stuck_below)
    n = max(len(prompts), 1)
    # `stuck` above is the DEPRECATED line-novelty metric, kept so numbers from
    # before 2026-09-25 stay comparable. It is biased LOW: _novel_tail returns
    # None for roughly half the generations (including the most degenerate ones,
    # which have no line structure at all) and this still divides by the full
    # prompt count. Re-scored on three arms it read 14.6% where the
    # compressibility metric read 27.1%.
    #
    # `degenerate` is the one to use. It scores every generation over 200 chars
    # and catches patterned repetition that is textually distinct line by line.
    ent = [_tail_entropy(o) for o in outs]
    n_ent = sum(1 for e in ent if e is not None)
    n_deg = sum(1 for e in ent if e is not None and e < DEGENERATE_BELOW)
    return {"stuck": n_stuck / n, "truncated": n_trunc / n,
            "degenerate": n_deg / max(n_ent, 1), "n_scored": n_ent,
            "stuck_scored": len(tails),
            "novel_tail": (sum(tails) / len(tails)) if tails else float("nan"),
            "n": n}
