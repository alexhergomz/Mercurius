"""stream_nf4.load_nf4 must equal from_pretrained + BitsAndBytesConfig.

Checked on the 4B, where both loaders fit in memory side by side. Also reports
the peak system memory each loader consumes, which is the point of the
streaming loader. A watchdog exits if MemAvailable falls below --floor-gb.

    python experiments/test_stream_nf4.py [--model models/qwen3.5-4b]
"""
import argparse
import os
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mercurius.guard import mem_available_gb
from mercurius.models.stream_nf4 import load_nf4
from mercurius.paths import BASE_MODEL, WIKITEXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(BASE_MODEL))
    ap.add_argument("--floor-gb", type=float, default=10.0)
    ap.add_argument("--skip-reference", action="store_true",
                    help="only stream-load and report ppl (for models whose "
                         "from_pretrained load does not fit)")
    a = ap.parse_args()
    low = [mem_available_gb()]

    def watch():
        while True:
            av = mem_available_gb()
            low[0] = min(low[0], av)
            if av < a.floor_gb:
                print(f"ABORT: MemAvailable {av:.1f} GiB < {a.floor_gb}", flush=True)
                os._exit(3)
            time.sleep(0.2)
    threading.Thread(target=watch, daemon=True).start()

    tok = AutoTokenizer.from_pretrained(a.model)
    x = tok(open(WIKITEXT).read()[:20000], return_tensors="pt").input_ids[:, :2048].cuda()

    @torch.no_grad()
    def evaluate(m):
        lg = m(input_ids=x, use_cache=False).logits[0].float()
        return lg, torch.nn.functional.cross_entropy(lg[:-1], x[0, 1:]).exp().item()

    base = mem_available_gb(); low[0] = base
    t0 = time.time()
    ms = load_nf4(a.model, keep_fp32=False)   # from_pretrained's dtype policy
    used_s = base - low[0]
    ls, ps = evaluate(ms)
    print(f"stream : {time.time() - t0:5.0f}s  peak use {used_s:5.1f} GiB  ppl {ps:.4f}",
          flush=True)
    if a.skip_reference:
        return 0
    del ms; torch.cuda.empty_cache(); time.sleep(3)

    base = mem_available_gb(); low[0] = base
    t0 = time.time()
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    mr = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                              device_map="cuda",
                                              quantization_config=q).eval()
    used_r = base - low[0]
    lr, pr = evaluate(mr)
    print(f"hf bnb : {time.time() - t0:5.0f}s  peak use {used_r:5.1f} GiB  ppl {pr:.4f}",
          flush=True)
    r = ((ls - lr).norm() / lr.norm()).item()
    ok = r == 0.0
    print(f"logits relL2 {r:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
