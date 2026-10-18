"""Distillation teachers. Every teacher returns the FINAL hidden states (post final norm, pre head) for the exact token
ids the student sees, plus its output head W_t, so the loss forms the teacher's log-probabilities chunk by chunk
(never a full (T, V) block). Any number of teachers can be combined (config: teachers, teacher_combine).

    hf      a transformers model in this process, frozen. model: a path / hub id, or "student" for the student's
            ORIGINAL weights (self-distillation; teacher and student may name the same checkpoint).
            load_in: bfloat16 | float16 | nf4 (bitsandbytes, streamed one tensor at a time, for large teachers).
    server  a llama.cpp server: llama-server -m <gguf> --embeddings --pooling none --embd-normalize -1
            -ub <longest window> --parallel 1. The head is dequantized from the same GGUF (`gguf`), or read from
            `head` (.npy / .pt). Frozen, no gradient, so a large MoE teacher never enters this process.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import torch

from mercurius.arch.common import trunk


class Teacher:
    name = "teacher"
    weight = 1.0
    head = None                           # (V, d_t), on device

    def hidden(self, ids):
        raise NotImplementedError


class HFTeacher(Teacher):
    def __init__(self, spec, student_model, device="cuda", log=print):
        path = student_model if spec["model"] == "student" else spec["model"]
        self.name = os.path.basename(str(path).rstrip("/")) + (" (student original)" if spec["model"] == "student"
                                                               else "")
        self.weight = float(spec["weight"])
        if spec["load_in"] == "nf4":
            from mercurius.legacy.models.stream_nf4 import load_nf4
            from mercurius.arch.model import resolve
            self.model = load_nf4(resolve(path)).eval()
        else:
            from mercurius.arch.model import load_hf
            self.model = load_hf(path, spec["load_in"], device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.head = self.model.get_output_embeddings().weight
        self._trunk = trunk(self.model)
        log(f"  teacher {self.name}: in-process ({spec['load_in']}), weight {self.weight:g}")

    @torch.no_grad()
    def hidden(self, ids):
        return self._trunk(input_ids=ids.view(1, -1)).last_hidden_state[0]


class ServerTeacher(Teacher):
    def __init__(self, spec, cache_dir, device="cuda", log=print, timeout=1800, patience=600):
        self.url = spec["url"].rstrip("/")
        self.weight = float(spec["weight"])
        self.name = f"server {self.url}"
        self.timeout, self.patience = timeout, patience
        self.max_len = spec.get("max_len")
        self.head = load_head(spec, cache_dir).to(device, torch.bfloat16)
        log(f"  teacher {self.name}: head {tuple(self.head.shape)} from {spec['gguf'] or spec['head']}, "
            f"weight {self.weight:g}")

    def _post(self, payload):
        req = urllib.request.Request(self.url + "/embeddings", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        deadline, attempt, last = time.time() + self.patience, 0, None
        while True:              # a teacher restart (model load) takes ~1-2 min: retry, do not crash the run
            try:
                return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            except urllib.error.HTTPError as e:
                last = e
                if 400 <= e.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
            if time.time() > deadline:
                break
            attempt += 1
            time.sleep(min(30, 2 * attempt))
        raise RuntimeError(f"teacher server {self.url}: {last}")

    @torch.no_grad()
    def hidden(self, ids):
        ids = [int(t) for t in ids.flatten().tolist()]
        d = self._post({"input": ids})
        d = d.get("data", [d]) if isinstance(d, dict) else d
        h = torch.tensor(d[0]["embedding"], dtype=torch.float32, device=self.head.device)
        if h.shape[0] != len(ids):
            raise RuntimeError(f"teacher returned {h.shape[0]} states for {len(ids)} ids (misaligned targets)")
        return h

    def check(self, probe_ids, max_len):
        """Fail at startup, not mid-run: raw (not L2-normalised) states, one per id, at the longest window."""
        probe = (list(probe_ids) * (max_len // max(len(probe_ids), 1) + 1))[:max_len]
        try:
            h = self.hidden(torch.tensor(probe))
        except RuntimeError as e:
            raise SystemExit(f"teacher server rejected a {max_len}-token window; restart it with -b / -ub >= "
                             f"{max_len}: {e}")
        n = h[:64].norm(dim=-1).mean().item()
        if abs(n - 1.0) < 0.05:
            raise SystemExit("teacher server returns L2-normalised embeddings: restart it with --embd-normalize -1")
        return n


def load_head(spec, cache_dir):
    """The teacher's output head as a (V, d) tensor: from a .npy / .pt file, or dequantized from the GGUF (cached)."""
    if spec.get("head"):
        p = spec["head"]
        if p.endswith(".npy"):
            import numpy as np
            return torch.from_numpy(np.load(p))
        return torch.load(p, map_location="cpu")
    os.makedirs(cache_dir, exist_ok=True)
    key = os.path.basename(spec["gguf"]).replace(".gguf", "")
    cached = os.path.join(cache_dir, f"head_{key}.pt")
    if os.path.exists(cached):
        return torch.load(cached, map_location="cpu")
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(spec["gguf"])
    by_name = {t.name: t for t in r.tensors}
    t = by_name.get("output.weight") or by_name.get("token_embd.weight")     # tied models have no output.weight
    if t is None:
        raise KeyError(f"{spec['gguf']}: no output.weight / token_embd.weight")
    rows = t.data.shape[0]                               # (V, bytes per row) -> (V, d), dequantized in slices
    W = torch.cat([torch.from_numpy(dequantize(t.data[i:i + 8192], t.tensor_type)).to(torch.bfloat16)
                   for i in range(0, rows, 8192)])
    torch.save(W, cached)
    return W


def build_teachers(cfg, out_dir, log=print):
    ts = []
    for spec in cfg.teachers:
        if spec["kind"] == "server":
            ts.append(ServerTeacher(spec, os.path.join(out_dir, "cache"), cfg.run.device, log))
        else:
            ts.append(HFTeacher(spec, cfg.student.model, cfg.run.device, log))
    tot = sum(t.weight for t in ts)
    for t in ts:
        t.weight /= tot
    return ts
