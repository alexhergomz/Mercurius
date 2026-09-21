"""The single-prefill RULER scorer must agree with the two-pass reference.

Reference: one teacher-forced forward over prompt+gold for the NLL, and
generate() from scratch for the prediction -- what ruler.score_sample did
before it reused the prompt's cache.

    python experiments/test_ruler_single_prefill.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mercurius.eval import ruler_gen as R
from mercurius.eval.ruler import score_sample, gold_value_mask
from mercurius.eval.retrieval_ab import build
from mercurius.paths import CACHE_DIR, CKPT_DIR, STAGE_AB

tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
m = build(str(CKPT_DIR / "adapters-4b27b-s8192-best.pt"), 512,
          str(CACHE_DIR / "kv_covs_4b.pt"), quantize=True)
ok = []
with torch.no_grad():
    for task in ("niah_single_3", "niah_multivalue", "niah_multiquery"):
        for s in R.generate(task, 4096, 2, tok):
            nll, nll_v, pred = score_sample(m, tok, s, -1)
            p = tok(s["input"] + s["answer_prefix"]).input_ids
            gids, _ = gold_value_mask(s["outputs"], tok)
            ids = torch.tensor([p + gids], device="cuda")
            lg = m(input_ids=ids, logits_to_keep=len(gids) + 1).logits[0, :-1].float()
            ref_nll = F.cross_entropy(lg, ids[0, len(p):]).item()
            g = m.generate(torch.tensor([p], device="cuda"), max_new_tokens=len(gids) + 16,
                           do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
            ref_pred = tok.decode(g[0, len(p):], skip_special_tokens=True)
            same = pred == ref_pred
            close = abs(nll - ref_nll) <= 0.02 + 0.02 * abs(ref_nll)
            ok.append(same and close)
            print(f"{task:<16} NLL {nll:.4f} vs {ref_nll:.4f}  pred equal {same}  "
                  f"[{'PASS' if ok[-1] else 'FAIL'}]")
            if not same:
                print(f"   new: {pred!r}\n   ref: {ref_pred!r}")
print(f"{sum(ok)}/{len(ok)} passed")
raise SystemExit(0 if all(ok) else 1)
