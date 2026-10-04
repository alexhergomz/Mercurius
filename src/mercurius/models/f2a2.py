"""F2A2 -- Free-For-All Attention: heads compete for each other, per token.

In standard MHA each head attends independently and the head outputs are mixed
by W_O with FIXED weights: once trained, head 3 contributes the same linear
share whether or not it has anything to say about this token. A head cannot
abstain and it cannot borrow. Call that the mute limit.

F2A2 runs a SECOND attention, across heads, SEPARATELY FOR EVERY TOKEN. At
token i there are H queries and H keys already lying around -- one per head --
so let them compete:

    S_i[h, g]    = q_h[i] @ W @ k_g[i]              (H x H, per token)
    alpha_i[h,:] = softmax_g S_i[h, :]              head h's query picks heads
    o_g[i]       = sum_j A_g[i, j] v_g[j]           the usual per-head output
    out_h[i]     = sum_g alpha_i[h, g] * o_g[i]     <- what actually gets used

The effective weight token i's head h places on (head g, position j) is
alpha_i[h,g] * A_g[i,j], which sums to one over the whole (head, position)
grid: a convex combination (over heads) of convex combinations (over
positions). No extra mass enters the residual stream.

THIS IS WHY IT IS FREE. Mixing the OUTPUTS, not the attention matrices, means
the per-head attention is untouched -- H ordinary calls that FlashAttention
serves as usual, no A materialised, no N^2 tensor. alpha is (B, N, H, H): its
cost is O(B N H^2 d), negligible beside the O(B H N^2 d) of the attention it
sits on top of. An earlier version of this file mixed the attention MATRICES
(out_h = sum_g alpha[h,g] A_g v_h), which applies a borrowed pattern to its own
values -- that needs the H x H cross product of patterns and values, forces A to
be materialised, and costs H times attention. Same convex-of-convex object,
strictly worse implementation; do not reintroduce it.

Note what the competition is computed from: q_h[i] and k_g[i], the query and key
of the SAME TOKEN in different heads. Not a summary, not the attended context,
not a pooled statistic -- the per-token vectors that already exist.

W is a single shared (d, d) bilinear form. Per-head forms would just be a
reparameterisation of the Q/K projections, which is not the point.

INIT MATTERS, and the obvious init is wrong. With W = I the score is q_h . k_g,
whose diagonal has no reason to dominate -- measured alpha_diag at init is 0.125
for H=8, i.e. exactly uniform. So the layer STARTS by handing every head the
average attention pattern: a smoothed, worse-than-MHA layer that must first
climb back to "keep your own pattern" before it can learn to borrow. A
comparison run from there partly measures that recovery.

Instead `diag_init` adds a learnable bias b to the diagonal of the score,
b = diag_init, so alpha starts at (near) the identity and the layer starts AT
MHA. Borrowing is then something the task has to PAY FOR in the loss, which is
the honest test: a mechanism that helps will move off the diagonal on its own,
and one that does not stays put and is visibly inert.

REDUCTION: with `no_mix=True`, alpha is the identity (each head keeps its own
pattern) and the layer is EXACTLY standard MHA, bit for bit. That is the
control -- a gain has to beat MHA itself, not a rescaled variant of it.

COST: one (B,N,H,H) score, one (B,N,H,H)x(B,N,H,d) matmul. At H=32, N=32768 that
is ~33M alpha entries per sequence and O(N H^2 d) FLOPs -- under a percent of the
attention it rides on. FlashAttention is used unchanged for the inner attention.

THIS IS ROADMAP 3.2, not a new idea. "Learned soft head assignment (free for
all, bounded)" already exists there, already cites talking-heads, and carries an
ordering note: item 1.4 (widen up_k so every query head gets its own key)
captures most of the same expressiveness at ZERO extra FLOPs, so 1.4 comes first
and this is the follow-up, not the alternative. Roadmap 3.2 also specifies a
cheaper form than the one built here -- a learned n_heads x n_kv mixing matrix on
the KEY side, 8 x 2 parameters per layer, initialised to the current hard GQA
assignment so GQA is the starting point. That respects the file-wide rule that
every change needs a function-preserving init; the version in this module starts
at MHA only when diag_init is large.

INTEGRATION: F2A2 GOES ON TOP OF PER-HEAD QUERY MAPS (item 1.4), and in this
model it is STRUCTURALLY INERT without them. Not a preference -- algebra.

MLA here kept GQA's 2 kv heads for 8 query heads, so the raw k_g takes only TWO
distinct values across the eight heads. The outer score S[h,g] = q_h W k_g then
has at most 2 distinct columns: alpha can express two preferences, not eight, and
the competition is rank-deficient before a single step is taken.

It gets worse if the obvious wiring is used. install_per_head_q applies
    q~_h = R_h^T q_h,      so the attention score is (R_h^T q_h) . c_j
                                                  = q_h . (R_h c_j)
i.e. head h's EFFECTIVE KEY is R_h c_j, per-head distinct even though the latent
c_j is shared. If F2A2 reads the post-R query and the shared k, then
    S[h, g] = q~_h W c
carries NO g dependence whatsoever, so alpha is exactly uniform over g by
construction and the layer is a no-op with extra FLOPs -- and it would LOOK like
a clean null result rather than a wiring bug.

So the key index must be formed as
    k_g := R_g c
and the per-head query maps are what make that object exist. Wire F2A2 to read
the per-head effective keys, never the shared latent.

WHAT THE SMOKE TEST CANNOT TELL YOU: experiments/test_f2a2.py builds plain MHA
with its own per-head k projection, so its keys are already distinct per head and
none of the above can arise there. A win in that test says the mechanism works
when it is handed distinct keys. It says nothing about whether the integration
handed it any.

RELATION TO PRIOR WORK, stated rather than discovered later:
  * Talking-Heads Attention (Shazeer et al., arXiv:2003.02436) is the closest
    relative: it mixes attention LOGITS across heads with learned LINEAR maps
    before and after the softmax. Two differences here. The mixing is CONVEX,
    so it is a distribution over heads and cannot amplify. And it is
    INPUT-DEPENDENT -- computed per token from that token's own q and k --
    where talking-heads applies the same learned matrix to every token.
  * Mixture-of-Heads / gated attention route a token to a SUBSET of heads,
    usually from the token alone. Here every head stays, and the competition is
    between heads' keys.

WHAT WOULD FALSIFY IT: if alpha converges to the identity, heads never borrow,
W_O already had the capacity, and the mechanism is inert. The diagonal mass of
alpha is therefore logged, not assumed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class F2A2Attention(nn.Module):
    def __init__(self, d_model, n_heads, bias=False, no_mix=False,
                 causal=True, diag_init=4.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.causal, self.no_mix = causal, no_mix
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)
        # identity init: the head-competition score starts as q_h . k_g
        self.W = nn.Parameter(torch.eye(self.d))                  # (d, d)
        self.head_scale = nn.Parameter(torch.zeros(n_heads))      # per-head temp
        # learnable diagonal bias: alpha starts ~identity, so the layer starts
        # at MHA and must be paid to borrow. diag_init=0 recovers the naive
        # uniform-at-init behaviour, for ablation.
        self.diag_bias = nn.Parameter(torch.full((n_heads,), float(diag_init)))
        self.register_buffer("_eye", torch.eye(n_heads), persistent=False)

    def forward(self, x, return_stats=False):
        B, N, _ = x.shape
        shape = lambda t: t.view(B, N, self.h, self.d).transpose(1, 2)
        q, k, v = shape(self.q(x)), shape(self.k(x)), shape(self.v(x))
        # INNER attention: ordinary, untouched, FlashAttention-eligible. SDPA
        # dispatches to the flash / mem-efficient kernel and never materialises
        # the N x N matrix.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        o_t = o.transpose(1, 2)                                    # (B,N,H,d)
        if self.no_mix:
            out, alpha = o_t, None                                 # alpha = identity
        else:
            # OUTER attention, PER TOKEN: at position i the H queries and H keys
            # already exist; let them compete. Head is the sequence axis here,
            # position is a batch axis -- so this is (B,N,H,H), never (B,H,N,N).
            q_t, k_t = q.transpose(1, 2), k.transpose(1, 2)        # (B,N,H,d)
            s = (q_t @ self.W) @ k_t.transpose(-2, -1)             # (B,N,H,H)
            s = s / math.sqrt(self.d) * self.head_scale.exp().view(1, 1, -1, 1)
            s = s + self.diag_bias.view(1, 1, -1, 1) * self._eye
            alpha = s.softmax(-1)                                  # over heads
            out = alpha @ o_t                                      # (B,N,H,d)
        out = self.o(out.reshape(B, N, self.h * self.d))
        if return_stats:
            st = {}
            if alpha is not None:
                a = alpha.detach()
                p = a.clamp_min(1e-9)
                # diag = how much each head keeps its OWN output. -> 1 means the
                # mechanism is inert and this is MHA with a rounding error.
                st.update(alpha_diag=float(a.diagonal(dim1=-2, dim2=-1).mean()),
                          alpha_entropy=float(-(p * p.log()).sum(-1).mean()),
                          alpha_row_sum_err=float((a.sum(-1) - 1).abs().max()),
                          uniform_entropy=math.log(self.h))
            return out, st
        return out


class MHAReference(nn.Module):
    """Plain MHA with the same parameter budget, as the control."""

    def __init__(self, d_model, n_heads, bias=False, causal=True):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.causal = causal
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, return_stats=False):
        B, N, _ = x.shape
        shape = lambda t: t.view(B, N, self.h, self.d).transpose(1, 2)
        q, k, v = shape(self.q(x)), shape(self.k(x)), shape(self.v(x))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        if self.causal:
            m = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
            att = att.masked_fill(m, float("-inf"))
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, N, self.h * self.d)
        out = self.o(o)
        return (out, {}) if return_stats else out
