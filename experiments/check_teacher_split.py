"""Does the patched llama-server (split pooling-none prompts, #65) give the teacher the
SAME distributions as single-micro-batch processing?

Reference (saved from the unpatched server, -ub 16384, one micro-batch):
  logs/teacher_ref12k_h.npy   12,000 tokens in one request
  logs/teacher_ref8k_h.npy    the first 8,000 of them, own request
The 8k-vs-12k difference on the shared 8,000 positions is the NOISE FLOOR of the
quantized MoE kernels (batch-shape dependent). The patched server must sit at that
floor, measured as the KL of the teacher's next-token distribution (teacher head),
which is what training consumes -- not raw hidden-state distance.

    .venv/bin/python experiments/check_teacher_split.py --url http://127.0.0.1:8078
"""
import argparse

import numpy as np
import torch

from mercurius.recovery.teacher_server import TeacherServer


def kl(ha, hb, W, pos):
    la = torch.log_softmax(ha[pos].cuda().float() @ W.T, -1)
    lb = torch.log_softmax(hb[pos].cuda().float() @ W.T, -1)
    k = (la.exp() * (la - lb)).sum(-1)
    top = (la.argmax(-1) == lb.argmax(-1)).float().mean()
    return float(k.mean()), float(k.max()), float(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8078")
    ap.add_argument("--long", type=int, default=32768)
    a = ap.parse_args()
    ids = np.load("logs/teacher_ref12k_ids.npy").tolist()
    r12 = torch.from_numpy(np.load("logs/teacher_ref12k_h.npy"))
    r8 = torch.from_numpy(np.load("logs/teacher_ref8k_h.npy"))
    W = torch.from_numpy(np.load("cache/teacher35b_head.npy")).cuda().float()
    g = torch.Generator().manual_seed(0)
    pos8 = torch.randint(0, 8000, (512,), generator=g)
    pos12 = torch.randint(0, 12000, (512,), generator=g)
    print("floor  ref 8k vs ref 12k (first 8000): KL mean %.5f max %.4f top1-agree %.4f"
          % kl(r8, r12, W, pos8), flush=True)
    t = TeacherServer(a.url)
    s12 = t.hidden(torch.tensor(ids), device="cpu", dtype=torch.float32)
    print("patched 12k (split) vs ref 12k:          KL mean %.5f max %.4f top1-agree %.4f"
          % kl(s12, r12, W, pos12), flush=True)
    s8 = t.hidden(torch.tensor(ids[:8000]), device="cpu", dtype=torch.float32)
    print("patched 8k vs ref 8k:                    KL mean %.5f max %.4f top1-agree %.4f"
          % kl(s8, r8, W, pos8), flush=True)
    # consecutive requests must not leak accumulated outputs between tasks
    s_short = t.hidden(torch.tensor(ids[:500]), device="cpu", dtype=torch.float32)
    print("short request rows:", s_short.shape[0], "(want 500)", flush=True)
    longids = (ids * (a.long // len(ids) + 1))[:a.long]
    sl = t.hidden(torch.tensor(longids), device="cpu", dtype=torch.float32)
    print(f"long {a.long} request rows: {sl.shape[0]} (want {a.long}); "
          f"first 12000 vs ref 12k: KL mean %.5f max %.4f top1-agree %.4f"
          % kl(sl[:12000], r12, W, pos12), flush=True)


if __name__ == "__main__":
    main()
