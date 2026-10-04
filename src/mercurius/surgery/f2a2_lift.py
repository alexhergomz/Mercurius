"""Install F2A2 on the real model: per-token competition between head OUTPUTS.

This is roadmap 3.2 applied to the 8 softmax-attention layers (the other 24 are
GDN-2 linear attention and have no softmax heads to mix). It REQUIRES per-head
query maps and raises without them, because the mechanism is provably inert
otherwise -- see below.

WHY IT MUST SIT ON TOP OF PER-HEAD-Q. The outer score is S[h,g] = q_h . W . k_g.
Our MLA kept GQA's 4 kv heads for 16 query heads, so the raw k is SHARED by every
head in a group of 4: within a group the score cannot vary with g at all, and
alpha is flat there by construction. Measured on the GQA proxy
(experiments/test_f2a2_gqa.py, 2026-09-25), the key-driven spread of alpha across
heads sharing a key was EXACTLY 0.00e+00 with shared keys and 2.35e-02 once
per-head maps existed. That zero is an algebraic identity, not a small number.

So head g's key here is its EFFECTIVE key R_g k, which is what per-head-q creates:
install_per_head_q applies q~_h = R_h^T q_h, so the attention score is
    (R_h^T q_h) . k  =  q_h . (R_h k)
Using RAW q against R_g k also makes the construction self-consistent: at W = I
the diagonal S[h,h] = q_h . R_h k is EXACTLY the real attention score of head h,
so alpha's diagonal is comparing like with like and diag_init genuinely starts the
layer at "keep your own output".

WHAT IS NOT CLAIMED. That the mechanism helps. The synthetic tests could not
answer that -- plain MHA already solves the channel task, so head specialisation
was never forced, and every arm landed inside the seed spread. The only way to
find out is to train it in this model against the same recipe without it. This
module exists so that experiment is possible, not because a toy predicted a win.

PLUMBING, and why it is done by stashing rather than by rewriting attention. The
stock forward computes attention internally and hands o_proj the concatenated head
outputs, so the head outputs are reachable at o_proj -- but q and k are not. Each
piece therefore stashes what it already computed:
  * PerHeadQ stashes the RAW per-head q (pre-R) and exposes R
  * LatentKV stashes k as it comes out of up_k
  * the o_proj wrapper consumes both, mixes, and calls the original o_proj
Nothing rewrites the attention forward, so FlashAttention/SDPA inside attention is
untouched and no N x N tensor is materialised. alpha is (B, N, H, H).

EXACT FUNCTION-PRESERVING INIT, AND CONVEXITY, BOTH BY CONSTRUCTION. The mask is
added to the head-wise score before the softmax:

    alpha = softmax(S - m(t) * offdiag),    m(t) = (1-tau)(30 + 2*max|S|)

alpha is therefore ALWAYS a softmax: rows sum to 1, entries >= 0, whatever the
weights do. Convexity is the shape of the object, not an invariant to police.
At tau = 0 every off-diagonal sits >= 30 below the diagonal so alpha is exactly
the identity; at tau = 1 the full competition runs. tau is ramped by the trainer
(--f2a2-anneal). W and head_scale are the entire mechanism -- no gate, no
sigmoid, no extra projection.

THREE EARLIER DESIGNS FAILED, each for a reason worth not repeating:

 1. A learnable bias on the score's DIAGONAL. Gives the identity only
    approximately and has to out-shout data of unknown scale. Swept on the real
    model: bias 24 PINNED the layer (diag gradient exactly 0, W's 6.9e-10, under
    Adam's eps=1e-8 where its scale-invariance fails), bias 6 perturbed the model
    by 5.8e-02 at init. Note the obvious explanation is wrong -- a vanishing
    softmax Jacobian makes gradients small, but Adam normalises per parameter, so
    3.8e-04 takes the same step as 3.6e-02. Underflow past eps is what pinned it.

 2. A zero-init gate on the output blend, out = o + g(borrowed - o) with
    g = s_h * sigmoid(W_g o). sigmoid bounded its own factor but s_h was FREE AND
    SIGNED, and at step 100 of the first f2a2 arm it had gone negative on several
    layers (layer 23 mean -0.00275) -- the output then extrapolates OUTSIDE the
    convex hull of the head outputs, the amplification this design forbids.
    Clamping s_h >= 0 fixes the sign and creates a dead parameter: measured,
    clamp_min(0) passes gradient 1 at exactly 0 and 0 once negative, so any head
    whose first gradient is negative is permanently shut.

 3. The annealed mask with a FIXED magnitude (30). Same flaw as (1), relocated:
    measured with W ~ N(0,0.3) and inflated q/k it left alpha_diag at 0.46
    instead of 1. Referencing the mask to 2*max|S| makes it unout-shoutable,
    since row differences are bounded by that -- verified exact with W ~ N(0,1),
    head_scale ~ N(0,1.5) and q/k scaled 10x.

CONSEQUENCE WORTH KNOWING: the anneal's effect is SHARP near tau = 1, because the
mask magnitude tracks the score span. alpha_diag stays ~1 until roughly tau=0.75
and then falls. Monotone and correct at both ends, but the adaptation is less
gradual than a linear ramp in tau suggests.

FALSIFICATION: at tau = 1 there is no gate to hide behind, so if borrowing is not
worth it the model must learn alpha -> I through the score itself. alpha_diag ~ 1
at tau = 1 is therefore genuine inertness and the model's own choice.
"""
import math

import torch
import torch.nn as nn


class _Stash:
    """Per-forward scratch shared between the wrapped pieces of one layer."""

    __slots__ = ("q_raw", "k", "R")

    def __init__(self):
        self.q_raw = self.k = self.R = None


class F2A2Mixer(nn.Module):
    """Wraps o_proj: mixes head outputs by a per-token H x H convex alpha."""

    def __init__(self, o_proj, stash, n_heads, head_dim, n_kv):
        super().__init__()
        self.o_proj = o_proj
        self._stash = stash
        self.h, self.d, self.n_kv = n_heads, head_dim, n_kv
        self.rep = n_heads // n_kv
        # o_proj is usually WRAPPED (VeRALinear has .base, not .weight), so go
        # through parameters() rather than assuming a bare nn.Linear.
        dev = next(o_proj.parameters()).device
        # W = I so the diagonal reduces to the real attention score exactly
        self.W = nn.Parameter(torch.eye(head_dim, device=dev, dtype=torch.float32))
        self.head_scale = nn.Parameter(torch.zeros(n_heads, device=dev,
                                                   dtype=torch.float32))
        # tau: the anneal variable, ramped 0 -> 1 by the trainer. It scales the
        # off-diagonal mask, so tau=0 is exactly the model F2A2 was installed on
        # and tau=1 is the full competition. See the module docstring for the
        # three designs this replaced and why each failed.
        self.register_buffer("tau", torch.zeros((), dtype=torch.float32),
                             persistent=False)
        # The mask lives on the OFF-DIAGONAL of the head-wise score. MASK_MAX=30
        # makes the init bit-exact in fp32: exp(-30) = 9.4e-14, so the
        # off-diagonal mass over 15 competitors is 1.4e-12, far under fp32 eps
        # 1.2e-7, and the diagonal rounds to exactly 1.0.
        self.register_buffer("_off", 1.0 - torch.eye(n_heads, device=dev),
                             persistent=False)
        self.MASK_MAX = 30.0
        self.last_stats = {}

    def forward(self, x):
        st = self._stash
        if st.q_raw is None or st.k is None or st.R is None:
            # No stash: nothing to mix. Happens if the layer was called without
            # the wrapped q/kv running first. Fail loudly rather than silently
            # degrade to plain attention, which would look like a null result.
            raise RuntimeError(
                "F2A2Mixer: q/k were not stashed for this layer. The per-head-q "
                "and latent-KV wrappers must run before o_proj.")
        B, N, _ = x.shape
        o = x.view(B, N, self.h, self.d)                       # head outputs
        q = st.q_raw.float()                                   # (B,N,H,d) RAW q
        k = st.k.float().view(B, N, self.n_kv, self.d)
        k = k.repeat_interleave(self.rep, dim=2)               # (B,N,H,d)
        # head g's EFFECTIVE key: R_g k. This is the whole reason per-head-q is
        # a precondition -- without R this tensor has no g dependence in a group.
        # R_g k, NOT R_g^T k. PerHeadQ computes q~_h = R_h^T q_h, so the real
        # score is q~_h . k = q_h . (R_h k) and head g's effective key is R_g k.
        # With the transpose the other way round the diagonal S[h,h] is NOT head
        # h's own attention score -- measured max|S[h,h] - real| = 8.03 against
        # 9.5e-07 for this form -- so diag_init would not start the layer at
        # "keep your own output" and the function-preserving init would be void.
        kk = torch.einsum("bnge,gde->bngd", k, st.R.float())
        s = torch.einsum("bnhd,de,bnge->bnhg", q, self.W, kk)
        s = s / math.sqrt(self.d) * self.head_scale.exp().view(1, 1, -1, 1)
        # THE ANNEAL: subtract m(t) from every off-diagonal entry, m large -> 0.
        # alpha stays a softmax throughout, so every row sums to 1 with all
        # entries >= 0 -- convexity is the SHAPE of the object, not a property to
        # maintain, and no weight can violate it. At m = MASK_MAX alpha is
        # exactly the identity; at m = 0 the full competition runs.
        # The mask is referenced to the ACTUAL score scale, not a constant. A
        # fixed magnitude can be out-shouted: measured with W ~ N(0,0.3) and
        # inflated q/k, a flat mask of 30 left alpha_diag at 0.46 instead of 1 --
        # the same "must out-shout data of unknown scale" flaw that made the
        # learnable diagonal bias fragile. Since row differences are bounded by
        # 2*max|s|, adding MASK_MAX + 2*max|s| guarantees every off-diagonal sits
        # at least MASK_MAX below the diagonal, for ANY weights.
        _span = s.detach().abs().amax() if s.numel() else s.new_zeros(())
        m_t = (1.0 - self.tau) * (self.MASK_MAX + 2.0 * _span)
        s = s - m_t * self._off
        alpha = s.softmax(-1)
        mixed = torch.einsum("bnhg,bngd->bnhd", alpha, o.float()).to(x.dtype)
        # Gate on grad, NOT on self.training: a freshly installed mixer defaults
        # to training=True even when the parent model was .eval()'d before
        # apply_f2a2, so keying off self.training silently logged nothing for the
        # whole eval and alpha_diag came back None. is_grad_enabled() is false
        # under no_grad, which is what an eval forward actually looks like.
        if not torch.is_grad_enabled():
            a = alpha.detach()
            ag = a.view(B, N, self.h, self.n_kv, self.rep)
            gi = torch.arange(self.h, device=a.device) // self.rep
            keep = torch.ones(self.h, self.n_kv, device=a.device, dtype=torch.bool)
            keep[torch.arange(self.h, device=a.device), gi] = False
            sp = ag.max(-1).values - ag.min(-1).values
            self.last_stats = {
                "tau": float(self.tau),
                "mask": float(m_t),
                # alpha_diag IS the falsification statistic now. The anneal forces
                # the mask to 0, so by the end the mechanism is fully ON and the
                # model has no gate to hide behind: if borrowing was never worth
                # it, the only way to decline is to learn alpha -> I through the
                # score. alpha_diag ~ 1 at tau = 1 therefore means genuinely
                # inert, and it is the model's choice rather than a schedule's.
                "alpha_diag": float(a.diagonal(dim1=-2, dim2=-1).mean()),
                # variation across heads that SHARE a key, excluding the group
                # holding h so diag_bias cannot account for it. 0 => the
                # competition has no information to act on.
                "key_driven_spread": float(sp[:, :, keep].mean()) if keep.any()
                else float("nan"),
            }
        st.q_raw = st.k = None                                 # consume
        return self.o_proj(mixed.reshape(B, N, self.h * self.d))


def _wrap_perheadq(phq, stash):
    """Make PerHeadQ stash the RAW per-head q and expose R, without changing it."""
    if getattr(phq, "_f2a2_wrapped", False):
        return
    base_forward = phq.forward
    n_heads, head_dim = phq.n_heads, phq.head_dim

    def forward(x):
        out = phq.base(x)
        s = out.shape[:-1]
        qg = out.view(*s, n_heads, 2 * head_dim)
        q, gate = qg.split(head_dim, dim=-1)
        stash.q_raw = q.detach() if not torch.is_grad_enabled() else q
        stash.R = phq.R
        q = torch.einsum("...hd,hde->...he", q.float(), phq.R).to(out.dtype)
        return torch.cat([q, gate], dim=-1).reshape(*s, -1)

    phq.forward = forward
    phq._f2a2_wrapped = True
    phq._f2a2_base_forward = base_forward


def _wrap_latentkv(kv, stash):
    if getattr(kv, "_f2a2_wrapped", False):
        return
    base_forward = kv.forward

    def forward(x):
        k, v = base_forward(x)
        stash.k = k
        return k, v

    kv.forward = forward
    kv._f2a2_wrapped = True


def apply_f2a2(model, verbose=True):
    """Install F2A2 on every softmax-attention layer that has per-head-q.

    Returns the list of installed mixers. Raises if per-head-q is absent, since
    F2A2 on shared keys is provably inert and would read as a clean null.
    """
    from mercurius.surgery.perhead_q import PerHeadQ
    from mercurius.surgery.transmla import LatentKV

    cfg = model.config.text_config if hasattr(model.config, "text_config") \
        else model.config
    trunk = model.model.language_model if hasattr(model.model, "language_model") \
        else model.model
    n_heads, head_dim = cfg.num_attention_heads, cfg.head_dim
    mixers, skipped = [], []
    for i, layer in enumerate(trunk.layers):
        sa = getattr(layer, "self_attn", None)
        if sa is None or not hasattr(sa, "q_proj"):
            continue
        if not isinstance(sa.q_proj, PerHeadQ):
            skipped.append(i)
            continue
        kv = None
        for attr in ("kv", "latent_kv", "kv_proj"):
            cand = getattr(sa, attr, None)
            if isinstance(cand, LatentKV):
                kv = cand
                break
        if kv is None:
            for m in sa.modules():
                if isinstance(m, LatentKV):
                    kv = m
                    break
        if kv is None:
            skipped.append(i)
            continue
        n_kv = kv.k_out // head_dim
        stash = _Stash()
        _wrap_perheadq(sa.q_proj, stash)
        _wrap_latentkv(kv, stash)
        sa.o_proj = F2A2Mixer(sa.o_proj, stash, n_heads, head_dim, n_kv)
        mixers.append(sa.o_proj)
    if not mixers:
        raise SystemExit(
            "--f2a2 found no layer with BOTH per-head query maps and a latent "
            "KV. F2A2 reading shared keys has exactly zero key-driven signal "
            "(measured 0.00e+00), so it would train to a null that looks like "
            "evidence. Pass --per-head-q, and note F2A2 only applies to the "
            "softmax layers -- the GDN-2 layers have no heads to mix.")
    if verbose:
        # count only the mixer's OWN parameters; o_proj's (and its adapter's)
        # already existed and are not new capacity.
        n_par = sum(p.numel() for m in mixers
                    for p in (m.W, m.head_scale,
                              ))
        print(f"  F2A2 on {len(mixers)} softmax attention layers "
              f"({n_heads} heads / {mixers[0].n_kv} kv, rep {mixers[0].rep}), "
              f"annealed off-diagonal mask so each head is EXACTLY itself at init; "
              f"{n_par:,} new params. Skipped {len(skipped)} layers with no "
              f"per-head-q or no latent KV (GDN-2 layers have no heads to mix).",
              flush=True)
    return mixers


@torch.no_grad()
def verify_identity(model, ids, mixers, tol=2e-2):
    """Compare gate=0 against the gate's CURRENT value. Named badly on purpose
    now that it is documented: it does NOT compare against the model without
    F2A2, because once installed that model is gone.

    So `ok` means "the current gate is close enough to 0 that the layer is still
    the model it was installed on" -- meaningful at init, and correctly False
    afterwards. Misread once already: called after deliberately opening the gates
    it returned ok=False and looked like a bug.

    The STRONGER check, and the one that actually validates the install, is to
    snapshot the logits BEFORE apply_f2a2 and compare after. That is what
    experiments logs/f2a2_gate_identity.log does, and it reads exactly 0.000e+00.
    """
    saved = [m.tau.detach().clone() for m in mixers]
    for m in mixers:                     # tau 0 -> mask maximal -> alpha == I
        m.tau.fill_(0.0)
    a = model(input_ids=ids).logits.float()
    for m, s_ in zip(mixers, saved):
        m.tau.copy_(s_)
    b = model(input_ids=ids).logits.float()
    rel = float((a - b).norm() / a.norm())
    ad = sum(m.last_stats.get("alpha_diag", float("nan")) for m in mixers) / len(mixers)
    return {"rel_diff_vs_identity": rel, "alpha_diag": ad, "ok": rel < tol}
