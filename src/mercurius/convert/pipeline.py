"""`mercurius convert`: the student checkpoint -> <out_dir>/converted, a self-contained model (arch/model.py).

Stages (exact unless noted), each checked against the original's logits on held-in text:
    1. fold norm gains into projections; folded norms -> ScaleNorm              exact
    2. Gated DeltaNet -> GDN-2 (q/k heads expanded to the value heads)          exact
    3. attention -> absorbable MLA, sequentially calibrated on the training mix  lossy (reported per layer)
       or, with MLA off, attention -> PlainAttention (the KV-quantization hook)  exact
The surgery runs in float32 (convert_dtype) and the result is saved in the student dtype.
"""
from __future__ import annotations

import os

import torch

from mercurius.arch.attention import PlainAttention
from mercurius.arch.common import trunk
from mercurius.arch.model import load_hf, load_tokenizer, save_model
from .lift import fold_norms, lift_gdn2, to_scalenorm
from .mla import convert_model


@torch.no_grad()
def _logits(model, ids):
    return model(input_ids=ids.view(1, -1)).logits[0].float()


def _rel(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def convert(cfg, log=print, convert_dtype="float32"):
    c = cfg.convert
    out = os.path.join(cfg.run.out_dir, "converted")
    dev = cfg.run.device
    tok = load_tokenizer(cfg.student.model, cfg.student.revision)
    model = load_hf(cfg.student.model, convert_dtype, dev, cfg.student.revision)
    from mercurius.train.data import TrainingMix
    mix = TrainingMix(cfg.data, tok, os.path.join(cfg.run.out_dir, "cache"), cfg.run.seed, log)
    probe = mix.window(max(c.verify_tokens, 16), torch.Generator().manual_seed(1)).to(dev)
    ref = _logits(model, probe) if c.verify_tokens else None

    def check(stage):
        if ref is None:
            return
        cur = _logits(model, probe)
        top1 = float((cur.argmax(-1) == ref.argmax(-1)).float().mean())
        log(f"  after {stage}: logits rel. L2 {_rel(ref, cur):.2e}, top-1 agreement {100 * top1:.1f}%")

    if c.fold_norms:
        fold_norms(model, log)
        if c.scalenorm:
            to_scalenorm(model, log)
    if c.gdn2.enabled:
        lift_gdn2(model, c.gdn2.expand_qk_heads, c.gdn2.seed_alpha, log)
    check("exact stages")
    m = c.mla
    if m.enabled:
        windows = mix.calibration_windows(m.calib_windows, m.calib_seq, m.calib_seed)
        k_rms = None if m.k_rms == "exact" else int(m.k_rms)
        convert_model(model, windows, m.budget, m.n_rope, m.rank_multiple, m.min_rank, c.per_head_q, k_rms, dev, log)
        check("MLA conversion")
    else:
        for i, layer in enumerate(trunk(model).layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "k_proj"):
                layer.self_attn = PlainAttention.from_hf(layer.self_attn, i)
        check("attention module swap")
    from mercurius.arch.model import DTYPES
    model.to(DTYPES[cfg.student.dtype])
    for mod in model.modules():                      # vectors that must stay fp32
        for n in ("A_log", "dt_bias", "q_map"):
            p = getattr(mod, n, None)
            if isinstance(p, torch.nn.Parameter):
                p.data = p.data.float()
    save_model(model, out, tok, extra={"source": cfg.student.model, "convert": dict(c)})
    log(f"  converted model -> {out}")
    return out
