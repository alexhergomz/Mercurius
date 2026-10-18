"""`mercurius export`: converted model + trained tensors -> a deployable checkpoint.

The VeRA adapters are merged into their bases and every projection is written in its QAT format (packed:
weight_packed / weight_scale for the 4-bit formats, weight_q8 / weight_scale for q8_0), so the exported weights are
bit for bit the weights the QAT forward used. q4_0 tensors use the compressed-tensors pack-quantized layout
(symmetric int4, group 32, code - 8), which HgRT and vLLM-class runtimes load directly; GGUF conversion is a
re-packing of the same codes and scales. The config records the format of every tensor, the KV-cache format the
model was trained for, and the cache size per token.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from mercurius.adapters import inject_vera, merge_vera
from mercurius.arch.common import trunk
from mercurius.arch.model import load_model, load_tokenizer, save_model
from mercurius.quant.qat import GATES
from mercurius.quant.weights import FORMATS


@torch.no_grad()
def build_trained(cfg, ckpt=None, log=print):
    """The converted model with its trained tensors loaded and the adapters merged (bf16, no QAT hooks)."""
    conv = os.path.join(cfg.run.out_dir, "converted")
    model = load_model(conv, cfg.student.dtype, cfg.run.device)
    a = cfg.train.adapters
    inject_vera(model, a.targets, a.rank, a.d_init, seed=0, log=lambda *_: None)
    ckpt = ckpt or os.path.join(cfg.run.out_dir, "ckpt", "trainable-final.pt")
    sd = torch.load(ckpt, map_location="cpu")
    own = dict(model.named_parameters())
    for k, v in sd.items():
        own[k].data = v.to(own[k].device)              # trained dtype (fp32 gains stay fp32)
    log(f"  loaded {len(sd)} trained tensors from {ckpt}")
    n = merge_vera(model)
    log(f"  merged {n} VeRA adapters")
    return model


def formats_for(model, q):
    """{module name: format} for every quantized tensor, following the QAT rules."""
    out = {}
    for i, layer in enumerate(trunk(model).layers):
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                full = f"{_prefix(model, layer)}.{name}"
                out[full] = q.gates if name.split(".")[-1] in GATES else q.weights
    emb = model.get_input_embeddings()
    for n, m in model.named_modules():
        if m is emb and q.embedding != "bf16":
            out[n] = q.embedding
        if m is model.get_output_embeddings() and m.weight.data_ptr() != emb.weight.data_ptr() and q.lm_head != "bf16":
            out[n] = q.lm_head
    return {k: v for k, v in out.items() if v != "bf16"}


def _prefix(model, layer):
    for n, m in model.named_modules():
        if m is layer:
            return n
    raise KeyError


@torch.no_grad()
def export(cfg, ckpt=None, log=print):
    model = build_trained(cfg, ckpt, log)
    tok = load_tokenizer(os.path.join(cfg.run.out_dir, "converted"))
    out = cfg.export.dir
    quant = {}
    sd = dict(model.state_dict())
    if cfg.export.weights == "qat":
        for name, fmt in formats_for(model, cfg.qat).items():
            w = sd.pop(name + ".weight")
            packed = _pack_rows(w, fmt)
            for k, v in packed.items():
                sd[f"{name}.{k}"] = v
            quant[name] = {"format": fmt, "shape": list(w.shape)}
    kv = dict(cfg.qat.kv) if cfg.qat.enabled else {"format": "none"}
    extra = {"quant": quant, "kv_cache": kv, "source": cfg.student.model}
    save_model(model, out, tok, extra=extra, state_dict=sd)
    spec = json.load(open(os.path.join(out, "config.json")))["mercurius"]
    bits = sum(FORMATS[f["format"]]["bits"] * _numel(f["shape"]) for f in quant.values())
    log(f"  exported -> {out}: {len(quant)} quantized tensors ({bits / 8 / 2**30:.2f} GiB), KV {kv['format']}, "
        f"{spec['kv_values_per_token']} cached values per token")
    return out


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _pack_rows(w, fmt, rows=4096):
    parts = [FORMATS[fmt]["pack"](w[i:i + rows]) for i in range(0, w.shape[0], rows)]
    return {k: torch.cat([p[k] for p in parts]).cpu() for k in parts[0]}
