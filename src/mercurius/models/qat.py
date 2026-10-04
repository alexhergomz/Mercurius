"""Quantization-aware training for the deployed format: 4-bit weights, 4-bit KV.

WHAT DEPLOYS (user, 2026-10-01): 4-bit weights and a 4-bit KV cache.

  weights   every matmul weight the deployed model holds, in its DEPLOYED form:
            VeRA merged into its base (W_nf4 + diag(b) B diag(d) A), the per-head
            query maps folded into q_proj, the latent rotation folded into the MLA
            down/up projections. Format NF4, blocksize 64, double-quantized
            statistics -- bitsandbytes' own kernels, so the fake-quantized weight
            IS the deployed weight, bit for bit. NF4 rather than a linear int4
            grid because the frozen base already sits on the NF4 grid (the student
            trains against an NF4 base): re-gridding to int4 would add a SECOND
            full quantization error on top, while NF4(W_nf4 + dW) only moves the
            weights dW actually pushes across a code boundary.
  gates     GDN-2's decay / erase / write gate projections (in_proj_a, _be, _bw)
            -- kept bf16 in training because the cumulative log-decay is
            numerically sensitive; deployed at --qat-gate-bits (4 = NF4, 8 = int8
            group 32, 16 = bf16). NOT small: the KDA / GDN-2 lifts made them
            channel-wise, 755 M parameters (in_proj_a 2560->4096 and be/bw
            2560->4096 each, x24) -- 0.39 GB at NF4, 0.80 GB at int8. Which one is
            measured (experiments/qat_ptq_sweep.py), not assumed.
  embedding tied with lm_head; quantized ONCE in place (frozen, no gradient).
  KV cache  the MLA latent c = W_down x, the only thing the attention layers
            cache. --qat-kv-quant tq (#66, user 2026-10-02): TurboQuant-MSE without
            QJL -- fixed random orthogonal rotation folded into down/up, per-token
            fp16 norm, Lloyd-Max codebook for the rotated unit vector's coordinate
            density (4.03 bits/value at 4 bits). Default "int": symmetric int4, groups of 32 along the latent, one fp16 scale
            per group (4.5 bits/value). Optional fixed orthogonal rotation of the
            latent (--qat-kv-rot orth), folded into down/up so it costs nothing at
            inference: whether it helps is measured, not assumed -- the latent
            is a whitened SVD basis with SORTED variances, which favours groups
            of neighbours as they are, while rotation mixes them to spread
            per-token outliers (QuaRot / SpinQuant).

GRADIENTS. A frozen base under a merged weight would normally make autograd form
the dense weight gradient g^T x for every layer -- as costly as the input
gradient, ~+50% backward. Not needed: the value is computed with the quantized
merged weight (detached), and the trainable part re-enters as a zero-valued
residual on x.detach():

    y = x Q(W)^T  +  f_theta(x.detach()) - f_theta(x.detach()).detach()

Value: exactly x Q(W)^T. Input gradient: Q(W)^T g, exact. Parameter gradient: the
straight-through estimator dL/dW = g^T x pushed through dW/dtheta, computed in
rank space by the same low-rank path training already uses.

Installs by rebinding instance forwards: no module is replaced and no parameter
is created or renamed, so resume files and adapter checkpoints load unchanged.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

GATES = ("in_proj_a", "in_proj_be", "in_proj_bw")


# ---------------------------------------------------------------- quantizers
@torch.no_grad()
def nf4_roundtrip(W, blocksize=64):
    """W -> the weight bitsandbytes' Linear4bit would hold (NF4, double quant)."""
    import bitsandbytes.functional as bnbf
    w = W.to(torch.bfloat16).contiguous()
    q, st = bnbf.quantize_4bit(w, blocksize=blocksize, quant_type="nf4",
                               compress_statistics=True)
    return bnbf.dequantize_4bit(q, st).to(W.dtype)


def int_group_roundtrip(x, bits=4, group=32):
    """Symmetric int quantization in groups of `group` along the last dim, one
    fp16 scale per group (absmax / qmax). Last dim padded to a multiple of group."""
    qmax = 2 ** (bits - 1) - 1
    d = x.shape[-1]
    pad = (-d) % group
    xf = x.float()
    if pad:
        xf = F.pad(xf, (0, pad))
    g = xf.reshape(*xf.shape[:-1], -1, group)
    s = (g.abs().amax(-1, keepdim=True) / qmax).half().float().clamp_min(1e-8)
    q = (g / s).round().clamp(-qmax, qmax) * s
    q = q.reshape(*xf.shape)[..., :d]
    return q.to(x.dtype)


_TQ_BOOKS = {}


def tq_codebook(r, bits):
    """Lloyd-Max codebook for ONE coordinate of a uniformly random unit vector in R^r.

    TurboQuant (Zandieh et al. 2025), MSE variant, no QJL: after a random rotation a
    unit vector's coordinates follow f(x) ~ (1 - x^2)^((r-3)/2) on [-1, 1] (-> N(0, 1/r)),
    independent of the data, so ONE scalar codebook optimal for that density is optimal
    for every coordinate. Returns (centroids, boundaries), fp32, cached per (r, bits)."""
    key = (int(r), int(bits))
    if key not in _TQ_BOOKS:
        x = torch.linspace(-1, 1, 400001, dtype=torch.float64)[1:-1]
        w = (1 - x * x).clamp_min(0) ** ((r - 3) / 2)
        w = w / w.sum()
        cdf = w.cumsum(0)
        L = 2 ** bits
        c = x[torch.searchsorted(cdf, (torch.arange(L, dtype=torch.float64) + 0.5) / L)]
        for _ in range(300):
            b = (c[1:] + c[:-1]) / 2
            idx = torch.bucketize(x, b)
            num = torch.zeros(L, dtype=torch.float64).index_add_(0, idx, w * x)
            den = torch.zeros(L, dtype=torch.float64).index_add_(0, idx, w)
            c = torch.where(den > 0, num / den.clamp_min(1e-300), c)
        _TQ_BOOKS[key] = (c.float(), ((c[1:] + c[:-1]) / 2).float())
    return _TQ_BOOKS[key]


def tq_roundtrip(c, bits):
    """TurboQuant-MSE on vectors c (..., r) ALREADY in the rotated basis: per-vector fp16
    norm, unit vector quantized coordinate-wise with the Beta Lloyd-Max codebook."""
    r = c.shape[-1]
    cen, bnd = tq_codebook(r, bits)
    cen, bnd = cen.to(c.device), bnd.to(c.device)
    cf = c.float()
    n = cf.norm(dim=-1, keepdim=True).half().float().clamp_min(1e-12)
    q = cen[torch.bucketize(cf / n, bnd)]
    return (q * n).to(c.dtype)


def ste(x, xq):
    """Value xq, gradient identity."""
    return x + (xq - x).detach()


# ---------------------------------------------------------------- weights
def _base_weight(lin):
    """Dense weight of a (possibly NF4) frozen Linear, fp32."""
    try:
        import bitsandbytes as bnb
        if isinstance(lin, bnb.nn.Linear4bit):
            return bnb.functional.dequantize_4bit(
                lin.weight.data, lin.weight.quant_state).float()
    except ImportError:
        pass
    return lin.weight.data.float()


def _vera_delta(m):
    I, O = m.base.in_features, m.base.out_features
    A = m.vera_A[:, :I].float() * m.vera_d.float().unsqueeze(1)
    B = m.vera_B[:O, :].float() * m.vera_b.float().unsqueeze(1)
    return B @ A


def _vera_adapter(m, x):
    """The trainable part of VeRALinear.forward (the low-rank path only)."""
    I, O = m.base.in_features, m.base.out_features
    A = m.vera_A[:, :I] * m.vera_d.unsqueeze(1)
    B = m.vera_B[:O, :] * m.vera_b.unsqueeze(1)
    return (x.to(A.dtype) @ A.T) @ B.T


def _quantize_weight(W, cfg, is_gate=False):
    if is_gate:
        b = cfg["gate_bits"]
        if b >= 16:
            return W
        return nf4_roundtrip(W) if b == 4 else int_group_roundtrip(W, b, 32)
    return nf4_roundtrip(W)


def _cached(mod, key, build):
    """No-grad (eval / generation): the deployed weight is fixed between optimizer
    steps, so build it once per parameter version instead of once per call -- decode
    otherwise re-merges and re-quantizes every weight for every token (a QAT run's
    eval took ~80 min instead of minutes). Training (grad on) drops the cache, so it
    never holds a second copy of the model while activations need the memory."""
    if torch.is_grad_enabled():
        mod.__dict__.pop("_qat_cache", None)
        return build()
    c = mod.__dict__.get("_qat_cache")
    if c is None or c[0] != key:
        c = (key, build())
        mod.__dict__["_qat_cache"] = c
    return c[1]


def release_qat_caches(model):
    """Drop every eval-time deployed-weight cache and return the memory (#68.4): ~9 GiB of
    dense quantized weights held after an eval, on top of the trainer's reserved memory and
    the teacher, exhausted unified memory at the QAT install of v3-absorb."""
    n = 0
    for m in model.modules():
        if m.__dict__.pop("_qat_cache", None) is not None:
            n += 1
    torch.cuda.empty_cache()
    return n


def _qat_vera_forward(self, x):
    cfg = self._qat

    def build():
        with torch.no_grad():
            W = _base_weight(self.base) + _vera_delta(self)
            return _quantize_weight(W, cfg, self._qat_gate).to(x.dtype)
    Wq = _cached(self, (self.vera_d._version, self.vera_b._version, x.dtype,
                        cfg["gate_bits"] if self._qat_gate else None), build)
    bias = self.base.bias
    y = F.linear(x, Wq, None if bias is None else bias.to(x.dtype))
    if not torch.is_grad_enabled():
        return y
    xd = x.detach()
    h = _vera_adapter(self, xd)
    return y + (h - h.detach()).to(y.dtype)


def _phq_weight(self):
    """q_proj in deployed form: (base + VeRA) with R_h^T folded into each head's
    query rows; gate rows untouched. fp32, differentiable in R / d / b."""
    inner = self.base
    if hasattr(inner, "vera_d"):
        W = _base_weight(inner.base) + _vera_delta(inner)
    else:
        W = _base_weight(inner)
    H, D = self.n_heads, self.head_dim
    W = W.view(H, 2 * D, -1)
    q, gate = W[:, :D], W[:, D:]
    q = torch.einsum("hde,hdi->hei", self.R.float(), q)    # R_h^T W_q,h
    return torch.cat([q, gate], 1).reshape(H * 2 * D, -1)


def _qat_phq_forward(self, x):
    inner = self.base
    key = (self.R._version, x.dtype) + ((inner.vera_d._version, inner.vera_b._version)
                                        if hasattr(inner, "vera_d") else ())

    def build():
        with torch.no_grad():
            return nf4_roundtrip(_phq_weight(self)).to(x.dtype)
    Wq = _cached(self, key, build)
    y = F.linear(x, Wq)
    if not torch.is_grad_enabled():
        return y
    xd = x.detach()
    f = self._qat_float_forward(xd)                       # theta-dependent path
    return y + (f - f.detach()).to(y.dtype)


# ---------------------------------------------------------------- MLA + KV
def _orth(r, seed):
    g = torch.Generator().manual_seed(seed)
    Q, Rr = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))
    return (Q * torch.sign(torch.diagonal(Rr))).float()


def _qat_latent_forward(self, x):
    cfg = self._qat

    # The rotation is NOT folded into the NF4 weights (#68): the whitened-SVD factors have
    # very unequal row / column scales (S in up, the inverse whitening in down), and mixing
    # them before blockwise NF4 broke NF4's per-block scaling -- latent error 2.2x vs 0.2
    # unrotated in a stress test. Weights are quantized as they are; the r x r rotation runs
    # in bf16 at write / read (r^2 MACs per token; in absorbed decode it folds into the
    # r-dim query).
    def build():
        return tuple(ste(w, nf4_roundtrip(w))
                     for w in (self.down.weight, self.up_k.weight, self.up_v.weight))
    Wd, Wk, Wv = _cached(self, (self.down.weight._version, self.up_k.weight._version,
                                self.up_v.weight._version), build)
    c = F.linear(x, Wd.to(x.dtype))
    R = self._qat_rot
    if cfg["kv_bits"] < 16:
        if cfg.get("kv_quant", "int") == "tq":
            Rr = R.to(c.device, c.dtype)
            c = ste(c, tq_roundtrip(c @ Rr.T, cfg["kv_bits"]) @ Rr)
        else:
            if R is not None:
                Rr = R.to(c.device, c.dtype)
                c = ste(c, int_group_roundtrip(c @ Rr.T, cfg["kv_bits"], cfg["kv_group"]) @ Rr)
            else:
                c = ste(c, int_group_roundtrip(c, cfg["kv_bits"], cfg["kv_group"]))
    return F.linear(c, Wk.to(c.dtype)), F.linear(c, Wv.to(c.dtype))


# ---------------------------------------------------------------- rotation record
def _tq_rotation_record(model, kv_bits, kv_quant, verbose=True, root="ckpt"):
    """The cache can only be decoded with the EXACT rotations and codebooks it was trained
    with, so they are part of the model (#68.2). Fixed, dense, Haar-random orthogonal
    rotations (QR of seeded Gaussians, sign-fixed): the exact TurboQuant assumption, any
    dimension (our latents are 390-656, not powers of two -- a randomized Hadamard would
    need padding, i.e. more cached coordinates), r^2 MACs per token = negligible.
    First install: write them (+ seeds + Lloyd-Max codebooks) to a content-keyed sidecar.
    Later installs (eval, deployment): regenerate, VERIFY bit-equality with the sidecar,
    and use the saved matrices -- a library change cannot silently alter them."""
    import hashlib
    import os
    from mercurius.surgery.transmla import LatentKV
    if kv_quant != "tq":
        return None
    lat = [(nm, mm) for nm, mm in model.named_modules() if isinstance(mm, LatentKV)
           and getattr(mm, "_qat_rot", None) is not None]
    rope = [(nm, mm) for nm, mm in model.named_modules()
            if getattr(mm, "_qat_R0", None) is not None]
    spec = [("lat", nm, int(mm._qat_rot.shape[0])) for nm, mm in lat] + \
           [("rope", nm, int(mm._qat_R0.shape[0])) for nm, mm in rope]
    key = hashlib.sha1(repr((spec, kv_bits)).encode()).hexdigest()[:10]
    path = os.path.join(root, f"tq_rotations_{key}.pt")
    cur = {f"{k}:{nm}": (mm._qat_rot if k == "lat" else mm._qat_R0).detach().float().cpu()
           for (k, nm, _), (_, mm) in zip(spec, lat + rope)}
    if os.path.exists(path):
        saved = torch.load(path, map_location="cpu")
        bad = [k for k in cur if not torch.equal(cur[k], saved["rot"][k])]
        if bad:
            print(f"  TQ rotations: {len(bad)} regenerated matrices differ from {path} "
                  f"(e.g. {bad[0]}) -- USING THE SAVED ONES", flush=True)
        for (k, nm, _), (_, mm) in zip(spec, lat + rope):
            R = saved["rot"][f"{k}:{nm}"]
            if k == "lat":
                mm._qat_rot = R.to(mm.down.weight.device)
            else:
                mm._qat_R0 = R.to(mm.k_proj.latent.k_rope.weight.device)
        if verbose:
            print(f"  TQ rotations loaded from {path} ({len(cur)} matrices, "
                  f"{'identical to regeneration' if not bad else 'regeneration differed'})",
                  flush=True)
    else:
        books = {}
        for _, _, r in spec:
            c, b = tq_codebook(r, kv_bits)
            books[f"{r}:{kv_bits}"] = (c.clone(), b.clone())
        os.makedirs(root, exist_ok=True)
        torch.save({"rot": cur, "spec": spec, "kv_bits": kv_bits, "codebooks": books,
                    "note": "TurboQuant-MSE rotations (Haar, seeded: latent 9000+i, "
                            "RoPE key 7000+layer_idx) and Lloyd-Max codebooks -- needed "
                            "to decode the cache (#68.2)"}, path)
        if verbose:
            print(f"  TQ rotations + codebooks SAVED to {path} ({len(cur)} matrices)",
                  flush=True)
    return path


# ---------------------------------------------------------------- install
@torch.no_grad()
def install_qat(model, kv_bits=4, kv_group=32, kv_rot="none", gate_bits=4,
                embed_bits=4, kv_quant="int", verbose=True):
    """Make every forward see the deployed 4-bit model. Idempotent."""
    from mercurius.adapters.lora import VeRALinear
    from mercurius.surgery.perhead_q import PerHeadQ
    from mercurius.surgery.transmla import LatentKV
    if kv_quant == "tq":
        kv_rot = "orth"           # TurboQuant's random rotation is part of the quantizer
    cfg = {"kv_bits": kv_bits, "kv_group": kv_group, "gate_bits": gate_bits,
           "kv_quant": kv_quant}
    n = {"vera": 0, "gate": 0, "phq": 0, "mla": 0}
    owned = set()
    for name, m in model.named_modules():
        if isinstance(m, PerHeadQ):
            owned.add(id(m.base))            # also on a repeated call: its VeRA is owned
        if isinstance(m, PerHeadQ) and not hasattr(m, "_qat"):
            m._qat = cfg
            m._qat_float_forward = types.MethodType(type(m).forward, m)
            m.forward = types.MethodType(_qat_phq_forward, m)
            n["phq"] += 1
    for name, m in model.named_modules():
        if isinstance(m, VeRALinear) and id(m) not in owned and not hasattr(m, "_qat"):
            m._qat = cfg
            m._qat_gate = any(name.endswith(g) for g in GATES)
            m.forward = types.MethodType(_qat_vera_forward, m)
            n["gate" if m._qat_gate else "vera"] += 1
    for li, (name, m) in enumerate((nm, mm) for nm, mm in model.named_modules()
                                   if isinstance(mm, LatentKV)):
        if hasattr(m, "_qat"):
            continue
        if getattr(m, "mol_struct", None) or getattr(m, "mol_routed", False) \
                or getattr(m, "blend", False):
            raise SystemExit(f"--qat supports plain MLA only ({name} is MoL/blend)")
        m._qat = cfg
        r = m.down.weight.shape[0]
        m._qat_rot = _orth(r, 9000 + li) if kv_rot == "orth" else None
        m.forward = types.MethodType(_qat_latent_forward, m)
        n["mla"] += 1
    for name, m in model.named_modules():
        if getattr(m, "_rope_decoupled", False):
            # absorbable MLA (#68): the decoupled RoPE key is cached too -> its projection
            # NF4, its cached values TurboQuant / int like the latent (in its forward).
            # Its rotation is created HERE (same seed rule as the forward's lazy path) so
            # it is recorded with the others.
            m._qat = cfg
            if kv_quant == "tq" and getattr(m, "_qat_R0", None) is None:
                d0 = m.k_proj.latent.k_rope.weight.shape[0]
                m._qat_R0 = _orth(d0, 7000 + int(m.layer_idx)).to(m.k_proj.latent.k_rope.weight.device)
    _tq_rotation_record(model, kv_bits, kv_quant, verbose)
    emb = model.get_input_embeddings()
    if embed_bits < 16 and not getattr(emb, "_qat_done", False):
        w = emb.weight.data
        q = nf4_roundtrip(w) if embed_bits == 4 else int_group_roundtrip(w, embed_bits, 32)
        err = ((q.float() - w.float()).norm() / w.float().norm()).item()
        emb.weight.data.copy_(q)
        emb._qat_done = True
        head = model.get_output_embeddings()
        if head is not None and head.weight.data_ptr() != emb.weight.data_ptr():
            # untied copy (a cast or a separate Parameter): it deploys tied, so
            # it must carry the same quantized values
            head.weight.data.copy_(q.to(head.weight.dtype))
            if verbose:
                print("  QAT: lm_head was a separate tensor -- copied the "
                      "quantized embedding into it", flush=True)
        if verbose:
            print(f"  QAT: embedding (tied lm_head) quantized in place to "
                  f"{'NF4' if embed_bits == 4 else f'int{embed_bits}'} "
                  f"(rel. error {err:.4f})", flush=True)
    if verbose:
        print(f"  QAT: NF4 weights on {n['vera']} VeRA layers + {n['phq']} q_proj "
              f"(per-head maps folded) + {n['mla']} MLA down/up; gates "
              f"{n['gate']} @ {'bf16' if gate_bits >= 16 else 'NF4' if gate_bits == 4 else f'int{gate_bits}'}; "
              + (f"KV latent TurboQuant-MSE {kv_bits}-bit (random rotation + Beta Lloyd-Max "
                 f"codebook + fp16 norm/token)" if kv_quant == "tq" else
                 f"KV latent int{kv_bits} g{kv_group} rot={kv_rot}"), flush=True)
    return n
