"""A multi-token forward on top of a cache must CONTINUE the recurrence.

  1. chunked prefill == single-pass prefill (last-position logits, and the
     next 16 greedy tokens decoded from each cache);
  2. scoring a continuation in one multi-token forward == scoring it one
     token at a time.

Stock transformers 5.6 fails (1) and (2) for Qwen3.5's linear layers: any
seq_len > 1 is treated as a fresh prefill from a zero state.

    python experiments/test_cache_continuation.py
"""
import copy
import torch
from mercurius.eval.retrieval_ab import build
from mercurius.paths import CACHE_DIR, CKPT_DIR, WIKITEXT
from transformers import AutoTokenizer
from mercurius.paths import STAGE_AB

tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
m = build(str(CKPT_DIR / "adapters-4b27b-s8192-best.pt"), 512,
          str(CACHE_DIR / "kv_covs_4b.pt"), quantize=True)
ids = tok(open(WIKITEXT).read()[:60000], return_tensors="pt").input_ids[:, :6000].cuda()
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
ok = []


def greedy(cache, first, n=16):
    out, nxt = [], first.argmax()
    for _ in range(n):
        out.append(int(nxt))
        o = m(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True)
        cache, nxt = o.past_key_values, o.logits[0, -1].argmax()
    return out


with torch.no_grad():
    one = m(input_ids=ids, use_cache=True, logits_to_keep=1)
    c = None
    for a, b in ((0, 1000), (1000, 1003), (1003, 4097), (4097, 6000)):  # includes a < k chunk
        o = m(input_ids=ids[:, a:b], past_key_values=c, use_cache=True, logits_to_keep=1)
        c = o.past_key_values
    r = rel(o.logits[0, -1], one.logits[0, -1])
    g1 = greedy(copy.deepcopy(one.past_key_values), one.logits[0, -1])
    g2 = greedy(c, o.logits[0, -1])
    ok.append(r < 3e-2 and g1 == g2)
    print(f"1. chunked (1000/3/3094/1903) vs single prefill: last logits relL2 {r:.2e}; "
          f"16 greedy tokens equal {g1 == g2}  [{'PASS' if ok[-1] else 'FAIL'}]")

    base = m(input_ids=ids[:, :5000], use_cache=True, logits_to_keep=1)
    cont = ids[:, 5000:5040]
    multi = m(input_ids=cont, past_key_values=copy.deepcopy(base.past_key_values),
              use_cache=True).logits[0].float()
    c, steps = copy.deepcopy(base.past_key_values), []
    for t in range(cont.shape[1]):
        o = m(input_ids=cont[:, t:t + 1], past_key_values=c, use_cache=True)
        c = o.past_key_values
        steps.append(o.logits[0, -1].float())
    r = rel(multi, torch.stack(steps))
    ok.append(r < 3e-2)
    print(f"2. 40-token continuation, one forward vs token by token: logits relL2 "
          f"{r:.2e}  [{'PASS' if ok[-1] else 'FAIL'}]")
print(f"{sum(ok)}/{len(ok)} passed")
raise SystemExit(0 if all(ok) else 1)
