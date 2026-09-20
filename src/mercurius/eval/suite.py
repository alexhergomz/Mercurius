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
def ce_and_topk(model, ids, n, ks=(1, 5, 10), chunk=1024, teacher=None):
    """Cross-entropy, top-k accuracy vs ground truth, and top-1/k agreement
    with a teacher if supplied. Chunked: vocab is 248,320."""
    x = ids[:n].unsqueeze(0).cuda()
    # Ask for hidden states and suppress the full logit tensor. The lm_head is
    # applied chunk-wise below instead, because at 8192 x 248,320 the full
    # logits are 4.1 GiB resident for the whole eval -- and two concurrent evals
    # in that state hard-reset this machine on 2026-09-20. Chunked, the peak is
    # one chunk's worth and it is freed each iteration.
    out = model(input_ids=x, output_hidden_states=True, logits_to_keep=1)
    h = out.hidden_states[-1][0]
    W_lm = model.get_output_embeddings().weight
    # hidden_states[-1] must be POST final-norm, or every logit is wrong. Checked
    # against the model's own last-position logits rather than assumed.
    _ref = out.logits[0, -1].float()
    _got = (h[-1:] @ W_lm.T).float()[0]
    if (_ref - _got).abs().max() > 1e-2 * _ref.abs().max().clamp_min(1e-6):
        raise RuntimeError(
            "hidden_states[-1] is not the tensor lm_head consumes; chunked "
            "logits would be wrong. Fall back to out.logits.")
    tgt = x[0, 1:]

    t_h = None
    if teacher is not None:
        t_out = teacher(input_ids=x, output_hidden_states=True,
                        logits_to_keep=1)
        t_h = t_out.hidden_states[-1][0]
        t_W = teacher.get_output_embeddings().weight

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
    del out, h
    if t_h is not None:
        del t_h, t_out
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
