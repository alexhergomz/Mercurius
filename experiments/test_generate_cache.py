"""Cached generation must match cache-free greedy decoding on the converted model.

RULER's EM scores model.generate(use_cache=True), which runs the GDN-2
recurrent kernel with carried conv/recurrent state and the MLA layers from a
KV cache -- paths the training loop never touches. The reference re-runs the
whole prefix at every step with no cache.

    python experiments/test_generate_cache.py [--adapters ckpt/adapters-...pt]
"""
import argparse
import torch
from transformers import AutoTokenizer
from mercurius.eval.retrieval_ab import build
from mercurius.eval import ruler_gen as R
from mercurius.paths import CACHE_DIR, CKPT_DIR, STAGE_AB

ap = argparse.ArgumentParser()
ap.add_argument("--adapters", default=str(CKPT_DIR / "adapters-4b27b-s8192-best.pt"))
ap.add_argument("--new", type=int, default=24)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
m = build(a.adapters, 512, str(CACHE_DIR / "kv_covs_4b.pt"), quantize=True)
ok = []
with torch.no_grad():
    for task, n in (("niah_single_1", 4096), ("niah_multivalue", 4096)):
        s = R.generate(task, n, 1, tok)[0]
        ids = torch.tensor([tok(s["input"] + s["answer_prefix"]).input_ids], device="cuda")
        g = m.generate(ids, max_new_tokens=a.new, do_sample=False, use_cache=True,
                       pad_token_id=tok.eos_token_id)[0, ids.shape[1]:]
        cur = ids
        for _ in range(a.new):
            nxt = m(input_ids=cur, use_cache=False, logits_to_keep=1).logits[0, -1].argmax()
            cur = torch.cat([cur, nxt.view(1, 1)], 1)
        ref = cur[0, ids.shape[1]:]
        same = int((g[:len(ref)] == ref[:len(g)]).cumprod(0).sum())
        ok.append(same == a.new)
        print(f"{task}@{n}: first {same}/{a.new} tokens identical  "
              f"[{'PASS' if ok[-1] else 'FAIL'}]\n  cached : {tok.decode(g)!r}\n"
              f"  no-cache: {tok.decode(ref)!r}\n  gold: {s['outputs']}")
raise SystemExit(0 if all(ok) else 1)
