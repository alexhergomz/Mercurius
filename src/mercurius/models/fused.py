"""torch.compile'd versions of the memory-bound elementwise pieces.

On this board (GB10, ~273 GB/s) an unfused fp32 elementwise chain over a
(T x hidden) tensor costs a full memory round trip per op. At 32k tokens:

    Qwen3_5RMSNorm, eager 15.2 ms -> compiled 1.5 ms, x81 per forward
    KDA/GDN-2 decay, eager 18.0 ms -> compiled 3.6 ms, x24 per forward

Only these small pure functions are compiled, never a whole model: the README
records that compiling the model lets inductor decompose SDPA and lose
FlashAttention. The decay is bitwise identical to eager; the norm differs by at
most one bf16 ULP (reduction order).

MERCURIUS_NO_COMPILE=1 turns all of it off.
"""
import os

import torch
import torch.nn.functional as F

ENABLED = os.environ.get("MERCURIUS_NO_COMPILE", "0") != "1"


def _rms(x, w, eps):
    o = x.float()
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    return (o * (1.0 + w.float())).type_as(x)


def _decay(a, A_log, dt_bias):
    a = a.unflatten(-1, (A_log.shape[0], A_log.shape[1]))
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


rms = torch.compile(_rms, dynamic=True) if ENABLED else _rms
decay = torch.compile(_decay, dynamic=True) if ENABLED else _decay


def patch_rmsnorm():
    """Route every Qwen3_5RMSNorm (zero-centred, (1 + w)) through the fused
    function. Idempotent. Returns whether it patched."""
    if not ENABLED:
        return False
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    if getattr(qm.Qwen3_5RMSNorm, "_mercurius_fused", False):
        return True
    qm.Qwen3_5RMSNorm.forward = lambda self, x: rms(x, self.weight, self.eps)
    qm.Qwen3_5RMSNorm._mercurius_fused = True
    return True
