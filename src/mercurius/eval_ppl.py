"""Perplexity of any checkpoint (original, converted or exported) on a text file, in non-overlapping windows."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def perplexity(model_dir, text_path, lengths=(2048, 8192), device="cuda", max_windows=8):
    from mercurius.arch.model import load_model, load_tokenizer
    from mercurius.arch.common import trunk
    model = load_model(model_dir, "bfloat16", device)
    tok = load_tokenizer(model_dir)
    ids = torch.tensor(tok(open(text_path, encoding="utf-8", errors="replace").read(),
                           add_special_tokens=False).input_ids)
    W = model.get_output_embeddings().weight
    tr = trunk(model)
    out = {}
    for L in lengths:
        nll, n = 0.0, 0
        for s in range(0, min(len(ids) - 1, max_windows * L), L):
            x = ids[s:s + L + 1].to(device)
            if len(x) < 2:
                break
            h = tr(input_ids=x[:-1].view(1, -1)).last_hidden_state[0]
            for i in range(0, h.shape[0], 1024):
                lg = (h[i:i + 1024] @ W.T).float()
                nll += float(torch.nn.functional.cross_entropy(lg, x[1 + i:1 + i + 1024], reduction="sum"))
            n += h.shape[0]
        out[L] = math.exp(nll / max(n, 1))
    return out
