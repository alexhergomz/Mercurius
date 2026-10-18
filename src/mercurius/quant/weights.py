"""Weight formats for quantization-aware training and export.

The first model trained against bitsandbytes NF4 (16 fixed non-uniform levels, blocks of 64, double-quantized fp8
scales). That format exists only in bitsandbytes: llama.cpp, vLLM, TensorRT-LLM, MLX and ONNX runtimes lack it, every
standard 4-bit format re-rounds NF4 weights (perplexity 3.390 NF4 vs 3.432 / 3.422 IQ4_NL / Q4_K), and its irregular
float levels defeat the int8 dot-product kernels decode relies on (docs/porting_findings.md #14). These formats are
the ones runtimes already run, so the QAT weight IS the deployed weight in any of them:

    q4_0    int4, symmetric, groups of 32 along the input dim, one fp16 scale (4.5 bits/weight). Rounding exactly as
            llama.cpp's Q4_0 (d = signed absmax / -8, q = trunc(x/d + 8.5) clamped to 15). Exports losslessly to GGUF
            Q4_0 and to compressed-tensors int4 (pack-quantized, symmetric, group 32), which vLLM / Marlin-class
            kernels and HgRT's dp4a GEMV run. The default.
    iq4_nl  llama.cpp IQ4_NL: the 16 non-uniform int8 levels below, groups of 32, fp16 scale (4.5 bits/weight).
            Better PTQ error than q4_0 on Gaussian-ish weights, still an int8-LUT format every llama.cpp backend runs.
    q8_0    int8, symmetric, groups of 32, fp16 scale (8.5 bits/weight) -- for sensitive projections.
    nf4     the legacy bitsandbytes format (needs bitsandbytes), kept to reproduce Mercurius-1.
    bf16    no quantization.

Each format provides fake(w) -> w rounded to the grid (same dtype), pack(w) -> tensors for export, and
unpack(...) -> w. Quantization is along the LAST dim (the input dim of a Linear weight), which must be a multiple of
32 for the groupwise formats -- the converter keeps every latent width a multiple of 64/128 for this reason
(docs/porting_findings.md #3, #15).
"""
from __future__ import annotations

import torch

GROUP = 32
IQ4NL_LEVELS = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                            dtype=torch.float32)


def _groups(w):
    if w.shape[-1] % GROUP:
        raise ValueError(f"input dim {w.shape[-1]} is not a multiple of {GROUP}")
    return w.float().reshape(*w.shape[:-1], w.shape[-1] // GROUP, GROUP)


# ------------------------------------------------------------------------------------------------ q4_0
def q4_0_codes(w):
    """-> (q uint8 in [0, 15] (..., G, 32), d fp16 (..., G)) with llama.cpp Q4_0 rounding."""
    g = _groups(w)
    idx = g.abs().argmax(-1, keepdim=True)
    d = (g.gather(-1, idx) / -8).squeeze(-1)
    d = d.half().float()                              # the stored scale is fp16
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.floor(g * inv.unsqueeze(-1) + 8.5).clamp(0, 15)
    return q.to(torch.uint8), d.half()


def q4_0_dequant(q, d):
    return (q.float() - 8) * d.float().unsqueeze(-1)


def q4_0_fake(w):
    q, d = q4_0_codes(w)
    return q4_0_dequant(q, d).reshape(w.shape).to(w.dtype)


def q4_0_pack(w):
    """compressed-tensors pack-quantized layout: weight_packed (N, K/8) int32, nibble i of word j = column 8j + i,
    stored code = q (signed value q - 8); weight_scale (N, K/32) fp16. GGUF Q4_0 stores the same codes and scales
    with a different nibble order (see export.py)."""
    q, d = q4_0_codes(w)
    N = w.shape[0]
    q = q.reshape(N, -1, 8).to(torch.int64)
    words = (q << (4 * torch.arange(8, device=q.device))).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return {"weight_packed": words.contiguous(), "weight_scale": d.reshape(N, -1).contiguous()}


def q4_0_unpack(t, shape):
    words = t["weight_packed"].to(torch.int64) & 0xFFFFFFFF
    q = (words.unsqueeze(-1) >> (4 * torch.arange(8, device=words.device))) & 15
    q = q.reshape(shape[0], -1, GROUP)
    return q4_0_dequant(q, t["weight_scale"]).reshape(shape)


# ------------------------------------------------------------------------------------------------ iq4_nl
def iq4_nl_codes(w, ntry=7):
    """Nearest-level codes plus a per-group scale searched like llama.cpp's quantize_row_iq4_nl_impl: map the
    signed extreme element onto either end level (-127 or +113) shifted by -ntry..ntry, refit d by least squares
    for each assignment, keep the lowest squared error. Any scale is a valid IQ4_NL block (the format stores
    d and 16 codes), so this search only affects accuracy, never compatibility."""
    g = _groups(w)
    lv = IQ4NL_LEVELS.to(g.device)
    bnd = (lv[1:] + lv[:-1]) / 2
    amax_i = g.abs().argmax(-1, keepdim=True)
    mx = g.gather(-1, amax_i).squeeze(-1)                                   # signed extreme
    best_err = torch.full(mx.shape, float("inf"), device=g.device)
    best_q = torch.zeros(g.shape, dtype=torch.uint8, device=g.device)
    best_d = torch.zeros_like(mx)
    for end, itry in [(e, i) for e in (-127.0, 113.0) for i in range(-ntry, ntry + 1)]:
        d0 = mx / (end + itry)                                              # candidate scale
        inv = torch.where(d0 != 0, 1.0 / d0, torch.zeros_like(d0)).unsqueeze(-1)
        q = torch.bucketize(g * inv, bnd)                                   # nearest level
        L = lv[q]
        sumqx, sumq2 = (L * g).sum(-1), (L * L).sum(-1)
        d = torch.where(sumq2 > 0, sumqx / sumq2, torch.zeros_like(sumqx)).half().float()
        err = ((d.unsqueeze(-1) * L - g) ** 2).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d, best_d)
        best_q = torch.where(better.unsqueeze(-1), q.to(torch.uint8), best_q)
    return best_q, best_d.half()


def iq4_nl_dequant(q, d):
    return IQ4NL_LEVELS.to(q.device)[q.long()] * d.float().unsqueeze(-1)


def iq4_nl_fake(w):
    q, d = iq4_nl_codes(w)
    return iq4_nl_dequant(q, d).reshape(w.shape).to(w.dtype)


def iq4_nl_pack(w):
    q, d = iq4_nl_codes(w)
    N = w.shape[0]
    q = q.reshape(N, -1, 8).to(torch.int64)
    words = (q << (4 * torch.arange(8, device=q.device))).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return {"weight_packed": words.contiguous(), "weight_scale": d.reshape(N, -1).contiguous()}


def iq4_nl_unpack(t, shape):
    words = t["weight_packed"].to(torch.int64) & 0xFFFFFFFF
    q = ((words.unsqueeze(-1) >> (4 * torch.arange(8, device=words.device))) & 15).reshape(shape[0], -1, GROUP)
    return iq4_nl_dequant(q, t["weight_scale"]).reshape(shape)


# ------------------------------------------------------------------------------------------------ q8_0
def q8_0_codes(w):
    g = _groups(w)
    d = (g.abs().amax(-1) / 127).half().float()
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d)).unsqueeze(-1)
    q = torch.round(g * inv).clamp(-127, 127).to(torch.int8)
    return q, d.half()


def q8_0_fake(w):
    q, d = q8_0_codes(w)
    return (q.float() * d.float().unsqueeze(-1)).reshape(w.shape).to(w.dtype)


def q8_0_pack(w):
    q, d = q8_0_codes(w)
    return {"weight_q8": q.reshape(w.shape[0], -1).contiguous(), "weight_scale": d.reshape(w.shape[0], -1).contiguous()}


def q8_0_unpack(t, shape):
    q = t["weight_q8"].float().reshape(shape[0], -1, GROUP)
    return (q * t["weight_scale"].float().unsqueeze(-1)).reshape(shape)


# ------------------------------------------------------------------------------------------------ nf4 (legacy)
def nf4_fake(w, blocksize=64):
    import bitsandbytes.functional as bnbf
    x = w.to(torch.bfloat16).contiguous()
    q, st = bnbf.quantize_4bit(x, blocksize=blocksize, quant_type="nf4", compress_statistics=True)
    return bnbf.dequantize_4bit(q, st).to(w.dtype)


def _no_pack(name):
    def f(*a, **k):
        raise NotImplementedError(f"{name} is not an export format; export with bf16 or a groupwise format")
    return f


FORMATS = {
    "q4_0": {"fake": q4_0_fake, "pack": q4_0_pack, "unpack": q4_0_unpack, "bits": 4.5},
    "iq4_nl": {"fake": iq4_nl_fake, "pack": iq4_nl_pack, "unpack": iq4_nl_unpack, "bits": 4.5},
    "q8_0": {"fake": q8_0_fake, "pack": q8_0_pack, "unpack": q8_0_unpack, "bits": 8.5},
    "nf4": {"fake": nf4_fake, "pack": _no_pack("nf4"), "unpack": _no_pack("nf4"), "bits": 4.127},
    "bf16": {"fake": lambda w: w, "pack": lambda w: {"weight": w.to(torch.bfloat16).contiguous()},
             "unpack": lambda t, shape: t["weight"].float().reshape(shape), "bits": 16.0},
}


def fake(w, fmt, rows=2048):
    """w rounded to format `fmt` (no gradient through the rounding), in row chunks so the fp32 temporaries of a
    large matrix stay small."""
    f = FORMATS[fmt]["fake"]
    with torch.no_grad():
        if fmt == "bf16":
            return w
        if w.dim() < 2 or w.shape[0] <= rows:
            return f(w)
        return torch.cat([f(w[i:i + rows]) for i in range(0, w.shape[0], rows)])


def ste(w, fmt):
    """Value fake(w), gradient identity (straight-through)."""
    if fmt == "bf16":
        return w
    return w + (fake(w, fmt) - w).detach()
