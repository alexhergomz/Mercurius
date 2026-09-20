"""Multi-token prediction: a gated dilated TCN over hidden states.

WHY NOT THE BUILT-IN HEAD. Qwen3.5-0.8B ships an MTP module, 20.45 M params:

    [ norm(h_t) ; norm(emb(x_{t+1})) ] -> fc(2048->1024)
        -> one full transformer layer -> norm -> lm_head

  * It consumes emb(x_{t+1}), so predicting t+2 requires having COMMITTED to
    t+1 -- a serial dependency, which is why it needs an inference engine that
    knows about it rather than just a forward pass.
  * mtp.layers.0 carries self_attn with k_proj/v_proj, so at decode it holds a
    SECOND KV cache at full width. This pipeline compresses the trunk's cache
    4x via MLA; the built-in head puts an uncompressed one straight back.
  * Depth 1.

ARCHITECTURE. A temporal convolutional stack with gated residual blocks:

    x = h
    for dilation D in (1, 2, 4):
        v = DWConv(x, k=3, dilation=D)             d*3
        g = DWConv(x, k=3, dilation=D)             d*3
        x = x + v * xatlu(g; alpha_D)              1   (trainable per block)
    delta = DWConv(x, d -> K*d, k=1, groups=d)     K*d
    z_j   = (h + delta_j) * (1 + g_j)              K*d
    logits_j = lm_head(z_j)          predicts t+1+j, j = 1..K

Receptive field 1 + 2*(1+2+4) = 15 positions. At d=1024, K=4: 26,627
parameters against the built-in head's 20.45 M -- 768x smaller.

WHY GATED, NOT A PLAIN CONV. A plain conv applies the same filter everywhere.
Gating makes the contribution content-dependent, which is the selectivity that
attention would otherwise provide and the reason gated convolutional sequence
models work at all. Over a 15-position horizon that is enough; this head is not
trying to replace attention, only to read the trajectory the trunk already
computed.

WHY xATLU. Huang, "Expanded Gating Ranges Improve Activation Functions",
arXiv:2405.20768. arctan as the gate gives a monotonically increasing first
derivative, and a trainable alpha per block expands the gate's range from (0,1)
to (-alpha, 1+alpha):

    gate(x)  = arctan(x)/pi + 0.5                  in (0, 1)
    xatlu(x) = x * (gate(x) * (1 + 2*alpha) - alpha)

The paper's own conclusion is that expanded gating ranges improve first-order
GLUs specifically, which is the block used here. alpha starts at 0, so the head
begins as a standard ATLU gate and widens only if training asks it to.

IDENTITY AT INIT, TWO WAYS.

(1) The head does NOT predict t+1. The trunk's own lm_head(h) keeps that,
    untouched, and the conv head predicts t+2 .. t+K+1. The primary prediction
    is therefore unaffected by construction rather than by numerics -- nothing
    about the head, trained or not, can perturb it.

(2) There is NO renormalization. An earlier version applied an rmsnorm to
    (h + delta), which looked harmless and was not: h is already the output of
    model.norm, whose zero-centred weight has mean(1+w) = 4.31, so
    rms(model.norm(x)) is about 4.36 rather than 1. Renormalizing rescaled the
    logits by ~0.23x AND discarded the per-channel shape lm_head is calibrated
    against. Now z_j = (h + delta_j) * (1 + g_j) with the output projection and
    g both at zero, so z_j = h EXACTLY and every head starts from the base
    next-token distribution -- a sensible prior for t+1+j.

    The lesson generalises: the check must compare against what the BASE model
    feeds lm_head, never against the new module's own output. Comparing a head
    to its own rmsnorm(h) passed at 4.77e-07 while the head was off by 4.4x.

The TCN internals get ordinary initialization and still receive gradient: the
output weight leaves zero first and opens the path, exactly as LoRA's zero B
does.

THE TARGET IS FREE. Head k at position t predicts token t+1+k. The teacher's
own base head at position t+k predicts that same token from more context, and
the recovery loop already computes teacher full-vocabulary logits at every
position for the main KL. Head k's target is those logits shifted by k -- no
extra teacher forward.

HONEST LIMITATION. Parallel heads cannot condition on their own intermediate
predictions, so accuracy falls off with depth faster than a sequential head's.
Gloeckle et al. (2404.19737) train parallel heads and report gains, so the
shape is viable; this trades per-position quality for one op and no cache.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class XATLU(nn.Module):
    """x * (arctan(x)/pi + 0.5) with a trainable expanded gating range."""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))     # 0 -> plain ATLU

    def forward(self, x):
        xf = x.float()
        gate = torch.atan(xf) / math.pi + 0.5
        a = self.alpha.float()
        return (xf * (gate * (1.0 + 2.0 * a) - a)).to(x.dtype)


class GatedTCNBlock(nn.Module):
    """Causal depthwise GLU block. Residual, so a zero value path is identity."""

    def __init__(self, d, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.value = nn.Conv1d(d, d, kernel, dilation=dilation, groups=d, bias=False)
        self.gate = nn.Conv1d(d, d, kernel, dilation=dilation, groups=d, bias=False)
        self.act = XATLU()

    def forward(self, x):                              # (B, d, T)
        xp = F.pad(x, (self.pad, 0))                   # causal
        return x + self.value(xp) * self.act(self.gate(xp))


class ConvMTPHead(nn.Module):
    def __init__(self, d_model=1024, k=4, kernel=3, dilations=(1, 2, 4),
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.d, self.k = d_model, k
        self.blocks = nn.ModuleList(
            [GatedTCNBlock(d_model, kernel, D) for D in dilations])
        # depthwise 1x1: channel c emits its own k values, K*d params not K*d^2
        self.out = nn.Conv1d(d_model, k * d_model, 1, groups=d_model, bias=False)
        nn.init.zeros_(self.out.weight)                # exact no-op at init
        self.gain = nn.Parameter(torch.zeros(k, d_model))
        self.receptive_field = 1 + sum((kernel - 1) * D for D in dilations)
        self.to(device=device, dtype=dtype)
        # fp32 like every other gain in this project: at 0.5 the bf16 ULP is
        # 3.91e-3 and an Adam step here is ~1e-4, which would round to zero
        self.gain.data = self.gain.data.float()
        for b in self.blocks:
            b.act.alpha.data = b.act.alpha.data.float()

    def forward(self, h):
        """h: (B,T,d) post-final-norm states -> (B,T,K,d); head j predicts t+1+j.

        No renormalization: h is already model.norm's output and carries the
        per-channel scale lm_head expects. At init z == h exactly.
        """
        B, T, d = h.shape
        x = h.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        delta = self.out(x).view(B, d, self.k, T).permute(0, 3, 2, 1)
        z = (h.unsqueeze(2) + delta).float() * (1.0 + self.gain)
        return z.to(h.dtype)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def mtp_targets(teacher_logits, j):
    """Target for head j (which predicts t+1+j): teacher logits at position t+j.

    Free -- the main KL already computed them. j starts at 1, because the trunk
    keeps t+1. The tail j positions have no target and are dropped by the
    caller.
    """
    return teacher_logits[:, j:] if teacher_logits.dim() == 3 else teacher_logits[j:]
