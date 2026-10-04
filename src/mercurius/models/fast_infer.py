"""Inference paths for the linear-attention layers: parallel where the math allows it.

Gated DeltaNet, KDA and GDN-2 are all linear recurrences, so ANY multi-token input --
a prompt, or a continuation on top of a cache -- runs as one call of fla's chunked
kernel (chunk_gated_delta_rule / chunk_kda / chunk_gdn2: intra-chunk matmuls, a scan
across chunks) seeded with the cached recurrent state; only true autoregressive decode
needs the single-step kernel (fused_recurrent_*). The causal conv is the same: a
multi-token conv over [cached K-1 raw frames, new frames], a single-token update
otherwise.

What stock transformers does instead, and what this fixes:
  * Qwen3_5GatedDeltaNet (the ORIGINAL arm's layers) treats every seq_len > 1 as a
    fresh prefill: zero initial state, zero-padded conv. A multi-token continuation
    therefore silently drops the whole prefix, which is why RULER scored gold spans
    one token at a time. Patched here with a continuation-aware forward (the same
    logic models/kda.py already has for our KDA / GDN-2 layers).
  * The single-token conv update stays PyTorch's (cuDNN) conv1d: fla's Triton
    causal_conv1d_update broke decoding on the real cache (see install below), and
    the op is negligible.
  * The stock multi-token conv: fla's Triton causal conv (as _fast_conv_for_stock).

install_fast_inference(model) sets model._supports_continuation = True when every
linear-attention layer can continue a cache with a multi-token forward; scoring code
(eval/ruler.py) then scores a span in ONE forward instead of token by token.
Numerics: identical math, different kernels -> bf16-rounding-level differences
(checked by experiments/check_fast_infer.py before use).
"""
import types

import torch
import torch.nn.functional as F


def fla_conv_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Drop-in for transformers' torch_causal_conv1d_update, on fla's Triton kernel.

    hidden_states (B, C, 1), conv_state (B, C, K) updated in place, weight (C, K)."""
    from fla.modules.convolution import causal_conv1d_update
    x = hidden_states.transpose(1, 2)                              # (B, 1, C)
    y, _ = causal_conv1d_update(x=x, cache=conv_state, weight=weight, bias=bias,
                                activation=activation)
    return y.transpose(1, 2)


def _stock_gdn_forward(self, hidden_states, cache_params=None, attention_mask=None):
    """Stock Qwen3_5GatedDeltaNet.forward + continuation of a cache by multi-token input."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    hidden_states = qm.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    has_prev = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
    single = has_prev and seq_len == 1
    continuing = has_prev and seq_len > 1

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if single:
        conv_state = cache_params.layers[self.layer_idx].conv_states
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias,
            self.activation)
    else:
        pad = 0
        if continuing:
            # last K-1 raw frames of the prefix, read BEFORE the cache is overwritten
            prev = cache_params.layers[self.layer_idx].conv_states[..., 1:].to(
                mixed_qkv.dtype).clone()
            cache_params.update_conv_state(mixed_qkv, self.layer_idx)
            mixed_qkv = torch.cat([prev, mixed_qkv], dim=-1)
            pad = prev.shape[-1]
        elif cache_params is not None:
            cache_params.update_conv_state(
                F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)),
                self.layer_idx)
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
            activation=self.activation, seq_idx=None)
        if pad:
            mixed_qkv = mixed_qkv[:, :, pad:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim],
                                    dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        rep = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    state = cache_params.layers[self.layer_idx].recurrent_states if has_prev else None
    if single:
        core, last = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
    else:
        # the stock cache keeps the state in bf16; the chunk kernel wants fp32
        core, last = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None if state is None else state.float(),
            output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True)
    if cache_params is not None:
        cache_params.update_recurrent_state(last, self.layer_idx)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(batch_size, seq_len, -1))


def install_fast_inference(model, verbose=True):
    """Parallel (chunked) kernels for every multi-token input, Triton conv updates for
    decode, continuation support on every linear-attention layer. In place."""
    from mercurius.models.kda import Qwen3_5KDAGatedDeltaNet, fla_causal_conv

    def conv(x, weight, bias=None, activation=None, seq_idx=None):
        return fla_causal_conv(x.transpose(1, 2), weight, bias,
                               activation=activation).transpose(1, 2)

    n_stock = n_ours = n_lin = 0
    for mod in model.modules():
        name = type(mod).__name__
        if not hasattr(mod, "recurrent_gated_delta_rule"):
            continue
        n_lin += 1
        # NOT fla_conv_update: it matches torch_causal_conv1d_update on a fresh
        # contiguous (B, C, K) cache (rel 2.9e-3) but MIS-COMPUTES on the model's
        # real cache -- decode collapsed ("286565670.<think><think>...", span NLL
        # 1.89 vs 0.089; experiments/check_fast_infer.py, 2026-10-02). The op is a
        # 4-tap depthwise conv per token, negligible beside the matmuls, so decode
        # keeps the PyTorch (cuDNN) update.
        if isinstance(mod, Qwen3_5KDAGatedDeltaNet):
            n_ours += 1                       # continuation already in kda.py
        elif name == "Qwen3_5GatedDeltaNet":
            mod.causal_conv1d_fn = conv
            mod.forward = types.MethodType(_stock_gdn_forward, mod)
            n_stock += 1
    model._supports_continuation = n_lin > 0 and n_stock + n_ours == n_lin
    if verbose:
        print(f"    fast inference: {n_ours} KDA/GDN-2 + {n_stock} stock GDN layers -- "
              f"chunked kernels for multi-token input (incl. cache continuation), "
              f"fused_recurrent for decode; continuation "
              f"{'ON' if model._supports_continuation else 'OFF'}", flush=True)
    return model
