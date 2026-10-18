"""Quantize the teacher to NF4 ONCE and save it, so training loads it pre-quantized.

On-the-fly bitsandbytes quantization of the 27B at every trainer start staged
enough memory that, on this shared 121 GB unified-memory board (vLLM holds
~42 GB), the kernel OOM-killed the trainer during the load on 2026-09-21. A
saved NF4 checkpoint is ~17 GB and loads without any bf16 staging.

A watchdog thread exits this process the moment system MemAvailable falls
below --floor-gb, so the load cannot take the machine down with it. Peak
MemAvailable consumption is reported, which is the number the trainer's
budget needs.
"""
import argparse
import os
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mercurius.guard import mem_available_gb
from mercurius.paths import MODELS_DIR, TEACHER_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(TEACHER_MODEL))
    ap.add_argument("--out", default=None)
    ap.add_argument("--floor-gb", type=float, default=10.0)
    a = ap.parse_args()
    out = a.out or str(MODELS_DIR / (os.path.basename(a.src.rstrip("/")) + "-nf4"))

    start = mem_available_gb()
    low = [start]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            av = mem_available_gb()
            low[0] = min(low[0], av)
            if av < a.floor_gb:
                print(f"\nABORT: MemAvailable {av:.1f} GiB < floor {a.floor_gb} GiB "
                      f"(started at {start:.1f}); exiting before the machine does",
                      flush=True)
                os._exit(3)
            time.sleep(0.2)
    threading.Thread(target=watch, daemon=True).start()
    print(f"MemAvailable at start {start:.1f} GiB; floor {a.floor_gb} GiB", flush=True)

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.bfloat16,
                                             device_map="cuda",
                                             quantization_config=q)
    print(f"loaded + quantized in {time.time() - t0:.0f}s; peak use "
          f"{start - low[0]:.1f} GiB of MemAvailable; CUDA allocated "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    m.save_pretrained(out)
    AutoTokenizer.from_pretrained(a.src).save_pretrained(out)
    stop.set()
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 2**30
    print(f"wrote {out} ({size:.1f} GiB); overall peak use {start - low[0]:.1f} GiB",
          flush=True)


if __name__ == "__main__":
    main()
