"""Qwen3_5KDAGatedDeltaNet — the converted layer as a real nn.Module.

Replaces the monkeypatch used for verification with a module that OWNS the
lifted parameters, so the conversion survives save/load and is trainable.

What changes versus the stock GDN layer:

    in_proj_a : Linear(hidden, H)     ->  Linear(hidden, H * Dk)
    A_log     : (H,)                  ->  (H, Dk)
    dt_bias   : (H,)                  ->  (H, Dk)
    decay     : g (B,T,H)             ->  g (B,T,H,Dk)
    kernel    : chunk_gated_delta_rule -> chunk_kda

Initialized by row-tiling, the layer reproduces GDN exactly (verified: the KDA
substitution moves the logits less than swapping GDN's own chunked kernel for
its own recurrent kernel).

Optional LoRA parameterization: keep the tiled projection frozen and learn a
zero-init low-rank delta on top. Exact at init because B = 0, and it costs
r*(hidden + H*Dk) instead of hidden*H*Dk -- ~4.7M rather than ~200M at 9B.

Note on head counts: A_log/dt_bias/in_proj_a are all sized by num_v_heads, and
q/k are repeat_interleave'd up to num_v_heads before the kernel. So the 9B case
(32 v-heads / 16 k-heads) needs no special handling; the diagonal is simply
num_v_heads x head_k_dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops import chunk_kda, fused_recurrent_kda
import transformers.models.qwen3_5.modeling_qwen3_5 as qm
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


class Qwen3_5KDAGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """GDN with per-channel (diagonal) decay — i.e. Kimi Delta Attention."""

    def __init__(self, config, layer_idx: int, lora_rank: int = 0):
        super().__init__(config, layer_idx)
        H, Dk = self.num_v_heads, self.head_k_dim
        self.kda_heads, self.kda_dim = H, Dk
        self.lora_rank = lora_rank
        # state passing: when not None, used as the kernel's initial_state and
        # refreshed (detached) each forward. Lets a short training segment start
        # from a state that only arises deep in a long context.
        self._carry = None        # recurrent state, read during forward only
        self._pending = None      # final recurrent state from the last forward
        self._conv_carry = None   # last (kernel-1) columns of the qkv stream
        self._conv_pending = None
        self._state_passing = False

        # scalar gate -> channel-wise gate
        self.in_proj_a = nn.Linear(self.hidden_size, H * Dk, bias=False)
        self.A_log = nn.Parameter(torch.zeros(H, Dk))
        self.dt_bias = nn.Parameter(torch.zeros(H, Dk))

        # Decay-tied positional phase (opt-in; None = off, and off is the
        # default so every existing checkpoint is unaffected).
        self.pe_c = None

        if lora_rank > 0:
            # zero-init delta: exact at init, trainable afterwards
            self.a_lora_A = nn.Parameter(torch.zeros(lora_rank, self.hidden_size))
            self.a_lora_B = nn.Parameter(torch.zeros(H * Dk, lora_rank))
            if not self.a_lora_A.is_meta:
                nn.init.normal_(self.a_lora_A, std=0.02)
        else:
            self.a_lora_A = self.a_lora_B = None

    # ------------------------------------------------------------------ init
    @classmethod
    @torch.no_grad()
    def from_gdn(cls, gdn: Qwen3_5GatedDeltaNet, config, layer_idx,
                 lora_rank: int = 0):
        """Function-preserving lift: row-tile the gate, copy everything else."""
        new = cls(config, layer_idx, lora_rank=lora_rank)
        H, Dk = new.kda_heads, new.kda_dim

        # everything that is not the decay gate transfers verbatim
        new.conv1d.load_state_dict(gdn.conv1d.state_dict())
        new.norm.load_state_dict(gdn.norm.state_dict())
        new.out_proj.load_state_dict(gdn.out_proj.state_dict())
        new.in_proj_qkv.load_state_dict(gdn.in_proj_qkv.state_dict())
        new.in_proj_z.load_state_dict(gdn.in_proj_z.state_dict())
        new.in_proj_b.load_state_dict(gdn.in_proj_b.state_dict())

        # the lift: each head's row repeated Dk times, each scalar broadcast
        new.in_proj_a.weight.data.copy_(
            gdn.in_proj_a.weight.data.repeat_interleave(Dk, dim=0))
        new.A_log.data.copy_(gdn.A_log.data.unsqueeze(-1).expand(H, Dk))
        new.dt_bias.data.copy_(gdn.dt_bias.data.unsqueeze(-1).expand(H, Dk))

        new.to(gdn.in_proj_a.weight.device, gdn.in_proj_a.weight.dtype)
        return new

    # ---------------------------------------------------------------- PE
    def enable_decay_phase(self, init: float = 0.0):
        """Turn on the decay-tied rotary phase. init=0 is the identity."""
        H = self.num_v_heads
        # NOT in_proj_a.weight: by the time this is called the projection may be
        # wrapped by a LoRA or VeRA adapter, whose wrapper holds no .weight of
        # its own (the base does). Any parameter of the module gives the device.
        dev = next(self.parameters()).device
        self.pe_c = nn.Parameter(torch.full((H,), float(init), device=dev,
                                            dtype=torch.float32))
        return self

    def _decay_phase(self, query, key, g):
        """Rotate q and k by an angle proportional to the CUMULATIVE log-decay.

        The eigenvalue becomes lambda_t = alpha_t^(1+ic): the modulus is
        unchanged, so the memory horizon is exactly as before, and the argument
        is c*log(alpha_t). Summing over positions, the kernel between m and t
        picks up a rotation by c*(G_t - G_m) where G is the running sum of
        log-decay -- a RELATIVE phase, decomposable into separate operations on
        q and k, which is the condition for compatibility with linear attention.

        Why tie it to the decay instead of giving position its own projection
        (as Selective RoPE, arXiv:2511.17388, does): the clock is then measured
        in nats of forgetting rather than in tokens. RoPE breaks on extrapolation
        because theta*t is indexed by token count, so a model trained at 4k has
        never seen phases past theta*4096 and needs NTK/YaRN rescaling. Here
        phase advances only as the model actually forgets, so there is no
        absolute frequency ladder pinned to a training length.

        The ladder is not lost, it is inherited: every channel has its own decay
        rate, so slow-decay channels rotate slowly (long-range position) and
        fast-decay channels rotate fast (local position). That is RoPE's
        frequency structure, arising from the decay spread rather than designed.

        Cost: one scalar per head, and a cumsum over a tensor the kernel already
        builds. c = 0 gives cos=1, sin=0 and returns q, k untouched.
        """
        if self.pe_c is None:
            return query, key
        G = g.float().cumsum(dim=1)                      # (B,T,H,Dk)
        # RoPE rotates channel PAIRS; one angle per pair, from the pair's mean
        # cumulative decay so both members share a frame.
        Gp = G.view(*G.shape[:-1], -1, 2).mean(-1)       # (B,T,H,Dk/2)
        th = Gp * self.pe_c.view(1, 1, -1, 1)
        cos, sin = th.cos(), th.sin()

        def rot(x):
            x2 = x.float().view(*x.shape[:-1], -1, 2)
            a, b = x2[..., 0], x2[..., 1]
            return torch.stack([a * cos - b * sin, a * sin + b * cos],
                               dim=-1).view_as(x).to(x.dtype)
        return rot(query), rot(key)

    # ------------------------------------------------------------- kernel
    def _run_kernel(self, query, key, value, g, hidden_states,
                    recurrent_state, want_final, single_step, cu_seqlens):
        """The delta-rule kernel call, isolated so subclasses can replace it.

        KDA drives the gated delta rule with a per-head SCALAR write strength
        beta. GDN-2 replaces that scalar with two channel-wise gates, which is a
        different kernel and a different signature -- but everything around it
        (the conv, the qkv split, the decay, state passing, the cache) is
        identical. Isolating the call is what lets the GDN-2 layer subclass this
        one instead of duplicating a 90-line forward that would then drift.
        """
        kernel = fused_recurrent_kda if single_step else chunk_kda
        return kernel(
            q=query, k=key, v=value, g=g.contiguous(),
            beta=self.in_proj_b(hidden_states).sigmoid(),
            initial_state=recurrent_state,
            output_final_state=want_final,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

    @torch.no_grad()
    def seed_decay_from_weights(self, strength: float = 1.0, width: float = 1.0,
                                descending: bool = False, source: str = "kconv",
                                generator=None, target: str = "A_log"):
        """Same decay ladder, PERMUTED per head and per layer.

        seed_decay_from_rope broadcasts ONE 128-rung ladder across all 16 heads
        of all 18 layers (`centered.unsqueeze(0)`), so 288 heads get bit-identical
        channel profiles. Measured: within-head channel std of the stored A_log is
        exactly 0, and the ladder is orthogonal to every weight statistic
        (|corr| < 0.01) -- the ORDERING carries no information.

        The ladder's SHAPE is already right and is kept untouched: it is linear in
        the channel index, so A_log is equispaced and the decay RATES are spread
        log-uniformly -- which is native KDA's actual prior (fla initializes
        dt_bias from a log-uniform draw over [0.001, 0.1]; it builds no RoPE
        ladder). Only the assignment of channels to rungs changes.

        Because this is a RANK map, the multiset of offsets is identical in every
        head -- so the head's total decay budget is preserved by construction, not
        merely in the mean, and the inherited per-head GDN scalar (which carries
        ~827x of real spread) is untouched.

        source:
          "kconv"  ||W_k[h,c]|| * |sum_t conv1d_k[h,c,t]|   (key-axis energy)
          "wk"     ||W_k[h,c]||
          "random" a random permutation of the SAME rungs -- the control that
                   separates "per-head diversity helps" from "weight-derived
                   assignment helps". Without it a win is uninterpretable.

        NOT exact. Neither is seed_decay_from_rope: GDN's function class IS the
        channel-uniform subspace, so any channel-differential init leaves it by
        definition. strength=0 recovers the bit-exact tiled point.
        """
        H, D = self.A_log.shape
        kd = H * D
        dev = self.A_log.device
        if source == "random":
            g = generator
            base = torch.rand(H, D, generator=g,
                              device=None if g is None else g.device)
            s = base.to(dev)
        else:
            Wk = self.in_proj_qkv.weight[kd:2 * kd].float().view(H, D, -1)
            s = Wk.norm(dim=-1)
            if source == "kconv":
                taps = self.conv1d.weight[kd:2 * kd, 0, :].float().view(H, D, -1)
                s = s * taps.sum(-1).abs()
            elif source != "wk":
                raise ValueError(f"unknown source {source!r}")
            # 7 key rows in this checkpoint have norm exactly 0 -> clamp before log
            s = s.clamp_min(1e-6).log()
        r = s.argsort(1).argsort(1).float() / max(D - 1, 1)      # ranks -> [0, 1]
        if descending:
            r = 1.0 - r
        ladder = width * (r - r.mean(dim=1, keepdim=True))       # zero-mean per head
        # target="dt_bias" is the parameter native KDA actually diversifies: fla
        # draws dt_bias log-uniform over [0.001, 0.1] per channel and leaves
        # A_log a per-head scalar. Our converted dt_bias is tiled from GDN's
        # per-head value and has within-head std 0.000000 -- so the axis native
        # KDA relies on carries no information here at all.
        # CAVEAT: rate = exp(A_log) * softplus(dt_bias). A zero-mean offset in
        # A_log preserves the head's geometric-mean rate exactly; in dt_bias it
        # does NOT, because softplus is nonlinear. Same assignment, weaker
        # invariant -- do not claim magnitude preservation for this target.
        p = {"A_log": self.A_log, "dt_bias": self.dt_bias}.get(target)
        if p is None:
            raise ValueError(f"unknown target {target!r}")
        p.data.add_((strength * ladder).to(p.dtype))
        return ladder

    def seed_decay_from_rope(self, rope_theta: float = 10_000_000.0,
                             strength: float = 1.0,
                             target_alpha: float | None = 0.60):
        """Seed channel decay across retention horizons mirroring RoPE bands.

        Trained GDN sits at median alpha ~0.969 (uniformly persistent) while
        natively-trained KDA reaches median ~0.471 with a wide spread. Tiled
        init starts every channel at GDN's persistent extreme -- far from where
        trained KDA lives. Spreading channels across horizons in the pattern
        RoPE used (fast-decaying <-> high frequency/local, slow-decaying <->
        low frequency/global) starts nearer the target distribution and gives
        the KDA layers the multi-scale positional structure the attention
        layers are about to lose.

        Applied to A_log, which enters as -exp(A_log): larger A_log -> faster decay.

        target_alpha shifts the whole distribution's median. Measured on this
        model with no training (ablate_decay.py), sweeping median alpha:

            alpha   ppl@32768   retrieval@4k
            0.90      21.764      12.492
            0.76      16.683      12.865   <- tiled+seeded default
            0.60      15.630      13.177   <- optimum
            0.47      15.838      13.388   <- dasc's trained-KDA figure
            0.30      16.691      13.764

        So the trained-KDA statistic is NOT a target to chase: 0.60 beats 0.47
        on perplexity. Shifting the median there is worth -6.3% ppl@32768 for
        free -- roughly 30x what 2.46M tokens of training moved it.

        Spread is a different story: 2x and 3x are clearly WORSE, so strength
        should stay near 1.0. Since alpha = exp(-exp(A_log)), a uniform shift d
        maps alpha -> alpha**exp(d), so one scalar sets the median.
        """
        Dk = self.kda_dim
        i = torch.arange(Dk, device=self.A_log.device, dtype=torch.float32)
        # RoPE frequency ladder over the channel axis, normalized to [0, 1]
        inv_freq = rope_theta ** (-2.0 * (i // 2) / Dk)
        band = (inv_freq.log() - inv_freq.log().min())
        band = band / band.max().clamp_min(1e-9)          # 0 = local, 1 = global
        # keep the head's mean log-decay, spread channels around it
        centered = (band - band.mean()) * strength
        self.A_log.data.add_(centered.unsqueeze(0).to(self.A_log.dtype))

        if target_alpha is not None:
            cur = float((-self.A_log.data.float().exp()).exp().median())
            if 0.0 < cur < 1.0 and 0.0 < target_alpha < 1.0:
                import math
                shift = math.log(math.log(target_alpha) / math.log(cur))
                self.A_log.data.add_(torch.tensor(
                    shift, dtype=self.A_log.dtype, device=self.A_log.device))

    def trainable_gate_parameters(self):
        """The parameters Stage B introduces, for the optimizer."""
        if self.lora_rank > 0:
            return [self.a_lora_A, self.a_lora_B, self.A_log, self.dt_bias]
        return [self.in_proj_a.weight, self.A_log, self.dt_bias]

    # --------------------------------------------------------------- forward
    def _decay(self, hidden_states):
        a = self.in_proj_a(hidden_states)
        if self.lora_rank > 0:
            a = a + F.linear(F.linear(hidden_states, self.a_lora_A), self.a_lora_B)
        a = a.unflatten(-1, (self.kda_heads, self.kda_dim))
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

    def _conv(self, mixed_qkv):
        """Causal depthwise conv + activation over (B, C, T), no cache."""
        T = mixed_qkv.shape[-1]
        if self.causal_conv1d_fn is not None:
            return self.causal_conv1d_fn(
                x=mixed_qkv, weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias, activation=self.activation, seq_idx=None)
        return F.silu(self.conv1d(mixed_qkv)[:, :, :T])

    # Mirrors the stock Qwen3_5GatedDeltaNet.forward of the installed
    # transformers (5.6) line for line, except for: the channel-wise decay, the
    # kernel call (isolated in _run_kernel), and segment state passing. The
    # previous version targeted an API (force_accelerate_hooks, cache
    # state_idx/record_past, module-level causal_conv1d_fn) that 5.6 does not
    # have, so it could not even be imported here.
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        hidden_states = qm.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state(self.layer_idx) and seq_len == 1)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        g = self._decay(hidden_states)                      # (B, T, H, Dk)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1),
                self.conv1d.bias, self.activation)
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv,
                                   (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(conv_state, self.layer_idx)
            # Carry the conv receptive field across segments. Without this the
            # causal Conv1d (kernel 4) sees zero padding at every segment
            # boundary instead of the previous tokens, corrupting the first few
            # q/k/v positions -- which then feed the recurrent state and compound.
            conv_pad = 0
            if self._state_passing:
                if self._conv_carry is not None:
                    mixed_qkv = torch.cat([self._conv_carry, mixed_qkv], dim=-1)
                    conv_pad = self._conv_carry.shape[-1]
                self._conv_pending = mixed_qkv[..., -(self.conv_kernel_size - 1):].detach()
            mixed_qkv = self._conv(mixed_qkv)
            if conv_pad:
                mixed_qkv = mixed_qkv[:, :, conv_pad:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        recurrent_state = (cache_params.layers[self.layer_idx].recurrent_states
                           if use_precomputed_states else None)
        if self._state_passing and self._carry is not None:
            recurrent_state = self._carry
        want_final = (cache_params is not None) or self._state_passing
        query, key = self._decay_phase(query, key, g)
        core_attn_out, last_recurrent_state = self._run_kernel(
            query, key, value, g, hidden_states,
            recurrent_state, want_final, use_precomputed_states, None)

        if self._state_passing and last_recurrent_state is not None:
            # Write to _pending, NOT _carry. Gradient checkpointing recomputes
            # the forward during backward; if the carried state changed in
            # between, the recompute diverges and torch raises CheckpointError.
            # promote_state() moves _pending -> _carry after the step.
            self._pending = last_recurrent_state.detach()
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


# ------------------------------------------------------------------ converter
def convert_to_kda(model, config, lora_rank: int = 0, seed_from_rope: bool = False,
                   copy_from_gdn: bool = True, verbose: bool = True):
    """Swap every GDN layer for its KDA lift, in place.

    copy_from_gdn=True   converting a loaded model -- tile the existing weights.
    copy_from_gdn=False  building a skeleton (e.g. on meta device) whose weights
                         a later load_state_dict will fill. Copying is invalid
                         there because meta tensors hold no data.
    """
    from mercurius.surgery.norm_fusion import get_trunk
    trunk = get_trunk(model)
    tcfg = getattr(config, "text_config", config)
    n = 0
    for i, layer in enumerate(trunk.layers):
        if not hasattr(layer, "linear_attn"):
            continue
        if copy_from_gdn:
            layer.linear_attn = Qwen3_5KDAGatedDeltaNet.from_gdn(
                layer.linear_attn, tcfg, i, lora_rank=lora_rank)
        else:
            layer.linear_attn = Qwen3_5KDAGatedDeltaNet(tcfg, i, lora_rank=lora_rank)
        if seed_from_rope and copy_from_gdn:
            theta = getattr(tcfg, "rope_parameters", {}) or {}
            layer.linear_attn.seed_decay_from_rope(
                float(theta.get("rope_theta", 1e7)) if isinstance(theta, dict) else 1e7)
        n += 1
    # record the conversion so it survives save/load
    tcfg.kda_lift = True
    tcfg.kda_lora_rank = lora_rank
    if verbose:
        print(f"converted {n} GDN layers -> KDA (lora_rank={lora_rank}, "
              f"rope_seeded={seed_from_rope})")
    return model


def load_kda_model(path, dtype=torch.float32, device="cuda"):
    """Load a converted checkpoint.

    `from_pretrained` cannot be used directly: it builds stock GDN modules with
    the ORIGINAL gate shapes (in_proj_a H x hidden, A_log (H,)) and then fails
    to load the lifted weights (H*Dk x hidden, (H, Dk)). The KDA modules must
    exist before the state dict is applied, so build the skeleton from config,
    convert, then load.
    """
    import os
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(path)
    tcfg = getattr(cfg, "text_config", cfg)
    if not getattr(tcfg, "kda_lift", False):
        raise ValueError(f"{path} is not a KDA-converted checkpoint "
                         "(config lacks kda_lift)")

    # Built directly ON THE DEVICE in the target dtype, with real
    # initialization (NOT meta + to_empty(): that leaves non-persistent buffers
    # like rotary_emb.inv_freq uninitialised -- measured relL2 1.18e-1).
    #
    # Then streamed in one tensor at a time. The previous version read every
    # shard into CPU memory (16.6 GiB of fp32 for the 4B), built a second full
    # fp32 copy on the CPU and only then moved it: a >33 GiB peak on a board
    # whose CPU and GPU share one 121 GiB pool with other services. The kernel
    # logged NVRM out-of-memory at exactly that point of a run on 2026-09-21.
    # Peak is now the model in its final dtype plus one tensor.
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        convert_to_kda(model, cfg, lora_rank=getattr(tcfg, "kda_lora_rank", 0),
                       copy_from_gdn=False, verbose=False)
    model.to(dtype)

    # The checkpoint may carry the VLM wrapper's tree (model.language_model.*)
    # while from_config on the saved TEXT config builds model.*; normalise. The
    # vision tower is dropped: LM-only surgery, reattached from the original.
    own = model.state_dict()
    loaded, dropped, unexpected = set(), 0, []
    shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    with torch.no_grad():
        for sh in shards:
            with safe_open(os.path.join(path, sh), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if ".visual." in k or k.startswith("visual."):
                        dropped += 1
                        continue
                    name = k.replace("model.language_model.", "model.")
                    if name not in own:
                        unexpected.append(name)
                        continue
                    own[name].copy_(f.get_tensor(k))
                    loaded.add(name)
    if dropped:
        print(f"  (dropped {dropped} vision-tower tensors; LM-only checkpoint)")
    # tied lm_head is expected to be absent from the shard; non-persistent
    # buffers never appear in state_dict at all
    missing = [k for k in own if k not in loaded and "lm_head" not in k]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]}")
    model.tie_weights()
    return model.eval()


# ------------------------------------------------------- state passing helpers
def enable_state_passing(model, on=True):
    """Carry each KDA layer's recurrent state across forward calls.

    Per "Understanding and Improving Length Generalization in Recurrent Models"
    (2507.02782): short-sequence training only explores a narrow attainable
    state distribution, which is why recurrent models fail to generalize to
    long context. Passing state across segments expands that distribution for
    ~10-15% extra tokens; models trained at 256 generalized to 8192 (32x).

    The full-attention layers have no state to carry, so under segmentation they
    see only the current window -- the short-window-attention design.
    """
    from mercurius.surgery.norm_fusion import get_trunk
    n = 0
    for l in get_trunk(model).layers:
        if hasattr(l, "linear_attn") and hasattr(l.linear_attn, "_state_passing"):
            l.linear_attn._state_passing = on
            l.linear_attn._carry = None
            l.linear_attn._conv_carry = None
            n += 1
    return n


def promote_state(model):
    """Move each layer's pending final state into the carry slot.

    Called AFTER backward, so the forward stays deterministic with respect to
    module state and remains compatible with gradient checkpointing.
    """
    from mercurius.surgery.norm_fusion import get_trunk
    for l in get_trunk(model).layers:
        la = getattr(l, "linear_attn", None)
        if la is None:
            continue
        if getattr(la, "_pending", None) is not None:
            la._carry = la._pending
            la._pending = None
        if getattr(la, "_conv_pending", None) is not None:
            la._conv_carry = la._conv_pending
            la._conv_pending = None


def reset_state(model):
    """Drop carried state -- call at document boundaries."""
    from mercurius.surgery.norm_fusion import get_trunk
    for l in get_trunk(model).layers:
        la = getattr(l, "linear_attn", None)
        if la is not None and hasattr(la, "_carry"):
            la._carry = None
            la._pending = None
            la._conv_carry = None
            la._conv_pending = None


def fuse_gate_lora(model, verbose=True):
    """Fold the in-class gate adapter into in_proj_a and switch it off.

    KDA carries its own rank-32 adapter on the decay projection:

        a = in_proj_a(x) + a_lora_B @ a_lora_A @ x

    stage_b_params treats these as EITHER/OR -- it returns the a_lora pair or
    in_proj_a.weight, never both. Training both at once (which --train-gate does)
    leaves a dense path and a low-rank path in parallel on the same projection,
    and measured on val-full the low-rank path wins: its update is 6.4-6.9x
    larger in norm than the dense one, and rank 1 where the dense delta is
    rank 10-44. The combined update is rank 1.

    That matters because in_proj_a is the TILED parameter. Stage B builds it by
    repeat_interleave, so it starts at exactly rank 16, and the point of training
    it densely is to let it leave that structure. It cannot do that while a
    dominant rank-1 path absorbs the gradient on the same output.

    Folding is exact: in_proj_a absorbs B@A, both factors are zeroed, and
    lora_rank is set to 0 so _decay skips the branch entirely. The decay path
    becomes one dense matrix with nothing constraining its rank.
    """
    n = 0
    for mod in model.modules():
        if getattr(mod, "lora_rank", 0) > 0 and getattr(mod, "a_lora_A", None) is not None:
            with torch.no_grad():
                delta = mod.a_lora_B.float() @ mod.a_lora_A.float()
                mod.in_proj_a.weight.data += delta.to(mod.in_proj_a.weight.dtype)
                mod.a_lora_A.zero_()
                mod.a_lora_B.zero_()
            mod.a_lora_A.requires_grad_(False)
            mod.a_lora_B.requires_grad_(False)
            mod.lora_rank = 0
            # in_proj_a.weight is now CHANGED and, once an adapter wraps it,
            # frozen -- so a checkpointer keyed on requires_grad drops it and
            # the fold is silently lost. Same tag the LoRA merge uses, read by
            # merged_base_names, for the same reason.
            mod.in_proj_a._absorbed_merge = True
            n += 1
    if verbose and n:
        print(f"  fused the in-class gate adapter into in_proj_a on {n} KDA "
              f"layers; the decay path is now a single dense matrix", flush=True)
    return n
