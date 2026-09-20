"""Qwen3_5GDN2GatedDeltaNet — Gated DeltaNet-2 as a function-preserving lift of KDA.

Gated DeltaNet-2 (Zhang et al., NVIDIA, arXiv:2605.22791; NVlabs/GatedDeltaNet-2)
replaces KDA's single scalar write-strength beta with TWO channel-wise gates:

    S_t = (I - k_t (b_t . k_t)^T) D_t S_{t-1} + k_t (w_t . v_t)^T

    b_t in [0,1]^{d_k}   erase gate, on the KEY axis
    w_t in [0,1]^{d_v}   write gate, on the VALUE axis
    D_t = Diag(alpha_t)  per-channel decay, unchanged from KDA

It reduces to KDA exactly when b_t = beta_t . 1_{d_k} and w_t = beta_t . 1_{d_v},
and to stock GDN when the decay also collapses to a scalar. Verified against
fla's own reference: with both gates tied to beta, the update becomes
v_new = beta (v - k^T h), which is the scalar gated delta rule.

WHY THIS ARCHITECTURE, HERE. The paper's stated target is "interference among
many compressed associations", and that is precisely and only where this
conversion is broken. Measured on RULER (docs/findings.md): single-needle
retrieval is fully preserved to 16k -- 100% EM, mean gold-span NLL 0.099 for the
original, the dense arm and the VeRA arm alike -- while multi-item retrieval
collapses, multiquery falling 97.5% -> 42.5% at 16k. One item is fine; several
interfere. A fixed-size recurrent state with a single scalar controlling both
erase and write cannot keep them apart, which is the deficiency GDN-2 names.

THE LIFT IS THE SAME SURGERY ALREADY DONE ONCE. KDA was produced from GDN by
row-tiling a per-head scalar gate into a per-channel one
(kda_model.from_gdn: in_proj_a.weight.repeat_interleave(Dk, dim=0)). GDN-2 is
that same operation applied to the OTHER gate: in_proj_b is Linear(hidden, H),
one scalar per head, and the two new projections are its rows repeated Dk and Dv
times. Identical at init because sigmoid is applied elementwise, so repeating
the pre-activation rows repeats beta exactly.

The tiled structure must then be free to break. Repeating a row makes every
channel in a head share one gate, which is the structure GDN-2 exists to escape;
trained with the whole projection free, or with a low-rank delta on top, the
channels can separate. This is the same requirement that applies to in_proj_a.
"""
import torch
import torch.nn as nn

from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2

from mercurius.models.kda import Qwen3_5KDAGatedDeltaNet


class Qwen3_5GDN2GatedDeltaNet(Qwen3_5KDAGatedDeltaNet):
    """KDA with the scalar write strength split into channel-wise erase/write."""

    def __init__(self, config, layer_idx: int, lora_rank: int = 0):
        super().__init__(config, layer_idx, lora_rank=lora_rank)
        H, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        self.gdn2_heads, self.gdn2_k, self.gdn2_v = H, Dk, Dv
        # in_proj_b stays, unused by the kernel but kept so a checkpoint written
        # before the lift still loads and so the reduction can be re-derived.
        self.in_proj_be = nn.Linear(self.hidden_size, H * Dk, bias=False)
        self.in_proj_bw = nn.Linear(self.hidden_size, H * Dv, bias=False)

    @classmethod
    @torch.no_grad()
    def from_kda(cls, kda: Qwen3_5KDAGatedDeltaNet, config, layer_idx,
                 lora_rank: int = 0):
        """Function-preserving lift: row-tile the scalar gate into both axes."""
        new = cls(config, layer_idx, lora_rank=lora_rank)
        H, Dk, Dv = new.gdn2_heads, new.gdn2_k, new.gdn2_v

        for name in ("conv1d", "norm", "out_proj", "in_proj_qkv", "in_proj_z",
                     "in_proj_b", "in_proj_a"):
            getattr(new, name).load_state_dict(getattr(kda, name).state_dict())
        new.A_log.data.copy_(kda.A_log.data)
        new.dt_bias.data.copy_(kda.dt_bias.data)
        if lora_rank > 0 and kda.a_lora_A is not None:
            new.a_lora_A.data.copy_(kda.a_lora_A.data)
            new.a_lora_B.data.copy_(kda.a_lora_B.data)

        # the lift: each head's single gate row, repeated across its channels
        w = kda.in_proj_b.weight.data
        new.in_proj_be.weight.data.copy_(w.repeat_interleave(Dk, dim=0))
        new.in_proj_bw.weight.data.copy_(w.repeat_interleave(Dv, dim=0))

        # in_proj_b is now DEAD: the GDN-2 kernel reads in_proj_be / in_proj_bw
        # instead, so nothing downstream consumes it and it receives no gradient.
        # It is kept, not deleted, because convert_to_gdn2 tiles the two gates
        # from it at load time -- it is the only record of what they started as.
        # But it must not sit in the optimizer: 294,912 parameters per model with
        # no gradient path still cost AdamW state and get reported as trainable,
        # which is how a run claims 4.71 M trainable when 4.42 M can move.
        new.in_proj_b.weight.requires_grad_(False)

        new.to(kda.in_proj_a.weight.device, kda.in_proj_a.weight.dtype)
        return new

    def _run_kernel(self, query, key, value, g, hidden_states,
                    recurrent_state, want_final, single_step, cu_seqlens):
        B, T = hidden_states.shape[:2]
        H, Dk, Dv = self.gdn2_heads, self.gdn2_k, self.gdn2_v
        b = self.in_proj_be(hidden_states).view(B, T, H, Dk).sigmoid()
        w = self.in_proj_bw(hidden_states).view(B, T, H, Dv).sigmoid()
        kernel = fused_recurrent_gdn2 if single_step else chunk_gdn2
        return kernel(
            q=query, k=key, v=value, g=g.contiguous(),
            b=b.contiguous(), w=w.contiguous(),
            initial_state=recurrent_state,
            output_final_state=want_final,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )


@torch.no_grad()
def convert_to_gdn2(model, verbose=True):
    """Replace every KDA layer with its GDN-2 lift, in place."""
    trunk = model.model.language_model if hasattr(model.model, "language_model") \
        else model.model
    n = 0
    for layer in trunk.layers:
        la = getattr(layer, "linear_attn", None)
        if not isinstance(la, Qwen3_5KDAGatedDeltaNet):
            continue
        layer.linear_attn = Qwen3_5GDN2GatedDeltaNet.from_kda(
            la, model.config.text_config if hasattr(model.config, "text_config")
            else model.config, la.layer_idx, lora_rank=getattr(la, "lora_rank", 0))
        n += 1
    if verbose:
        H = 0
        for layer in trunk.layers:
            la = getattr(layer, "linear_attn", None)
            if isinstance(la, Qwen3_5GDN2GatedDeltaNet):
                H += (la.in_proj_be.weight.numel() + la.in_proj_bw.weight.numel())
        print(f"  GDN-2: lifted {n} KDA layers; +{H/1e6:.2f} M gate parameters "
              f"(tiled from in_proj_b, exact at init)", flush=True)
    return n
