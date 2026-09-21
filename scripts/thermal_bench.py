"""Energy per token of the dominant workload, at whatever GPU clock is set.

Under a thermal ceiling, sustained throughput ~ (heat the cooler removes) /
(joules per token). Pacing (pauses) changes neither, so it cannot beat that
ceiling; a lower locked clock can, because voltage falls with frequency and
dynamic power goes ~ f V^2 while work per cycle stays fixed.

Measures the 27B NF4 teacher forward at 8192 tokens (~60% of a training
step's FLOPs) in short bursts: cool to --start-c, run until --stop-c or
--burst-s, sampling NVML power at 20 Hz. Reports tokens/s while running, mean
power, joules/token, and heating rate. Run it once per clock setting, e.g.

    sudo nvidia-smi -lgc 0,2000     # lock SM clock max (needs root)
    python scripts/thermal_bench.py --tag 2000
    sudo nvidia-smi -rgc            # restore default

Nothing here changes a clock; it only measures.
"""
import argparse
import json
import threading
import time

import pynvml
import torch

from mercurius.guard import acpi_max_c, cap_cuda_memory
from mercurius.models.stream_nf4 import load_nf4
from mercurius.paths import LOGS_DIR, TEACHER_MODEL
from mercurius.surgery.norm_fusion import get_trunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--bursts", type=int, default=3)
    ap.add_argument("--start-c", type=float, default=60.0)
    ap.add_argument("--stop-c", type=float, default=82.0)
    ap.add_argument("--burst-s", type=float, default=60.0)
    a = ap.parse_args()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    temp = lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
    cap_cuda_memory(40)
    m = load_nf4(str(TEACHER_MODEL), verbose=False).eval()
    trunk = get_trunk(m)
    x = torch.randint(1000, 100000, (1, a.seq), device="cuda")
    with torch.no_grad():                      # warm-up (kernel selection)
        trunk(input_ids=x[:, :1024])
    torch.cuda.synchronize()

    rows = []
    for b in range(a.bursts):
        while temp() > a.start_c or acpi_max_c() > a.start_c + 15:
            time.sleep(5)
        samples, stop = [], threading.Event()

        def sampler():
            while not stop.is_set():
                samples.append((time.time(), pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                                temp(), pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)))
                time.sleep(0.05)
        th = threading.Thread(target=sampler, daemon=True)
        t_start, T0, n = time.time(), temp(), 0
        th.start()
        with torch.no_grad():
            while time.time() - t_start < a.burst_s and temp() < a.stop_c:
                trunk(input_ids=x)
                torch.cuda.synchronize()
                n += 1
        dt = time.time() - t_start
        stop.set(); th.join()
        P = sum(s[1] for s in samples) / len(samples)
        clk = sum(s[3] for s in samples) / len(samples)
        row = {"tag": a.tag, "burst": b, "forwards": n, "seconds": dt,
               "tok_s": n * a.seq / dt, "mean_W": P, "J_per_tok": P * dt / (n * a.seq),
               "mean_sm_MHz": clk, "T_start": T0, "T_end": temp(),
               "heat_C_per_s": (temp() - T0) / dt}
        rows.append(row)
        print(f"  burst {b}: {n} fwd in {dt:5.1f}s  {row['tok_s']:7.0f} tok/s  "
              f"{P:5.1f} W  {row['J_per_tok'] * 1e3:6.3f} mJ/tok  SM {clk:5.0f} MHz  "
              f"{T0:.0f}->{row['T_end']:.0f}C ({row['heat_C_per_s']:+.2f} C/s)", flush=True)
    out = LOGS_DIR / "thermal_bench.jsonl"
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
