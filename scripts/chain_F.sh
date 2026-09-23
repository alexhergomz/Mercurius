#!/usr/bin/env bash
# RULER on run E, then the bf16 teacher's perplexity (the honest reference for
# "the student beats its teacher": every teacher number so far is NF4).
cd "$(dirname "$0")/.."
scripts/guarded.sh logs/ruler-E.log python -m mercurius.eval.ruler \
  --arms E150=ckpt/adapters-4b27b-E150-best.pt --mla-groups cache/mla_groups_retr_4096.json \
  --lengths 4096 8192 16384 32768 --samples 10 --quantize --out logs/ruler_E_4k_32k.json
echo "ruler E exit $?"
scripts/guarded.sh logs/teacher-bf16-ppl.log python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius import guard
from mercurius.eval.suite import ce_and_topk, report
from mercurius.paths import BASE_MODEL, TEACHER_MODEL, WIKITEXT
guard.cap_cuda_memory(90)
cool = lambda: guard.cool_to(75, 85)
tok = AutoTokenizer.from_pretrained(str(BASE_MODEL))
ids = tok(open(WIKITEXT).read(), return_tensors="pt").input_ids[0]
m = AutoModelForCausalLM.from_pretrained(str(TEACHER_MODEL), dtype=torch.bfloat16,
                                         device_map="cuda").eval()
with torch.no_grad():
    for n in (2048, 8192):
        report(f"teacher 27B bf16 @{n}", ce_and_topk(m, ids, n, before_forward=cool))
PY
echo "teacher bf16 exit $?"
