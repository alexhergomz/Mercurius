"""Does F2A2 do anything in OUR architecture -- GQA-style shared keys?

experiments/test_f2a2.py answered the wrong question. It built plain MHA, where
every head owns its own k projection, so F2A2 was handed per-head-distinct keys
for free. Our model does not look like that: MLA kept GQA's 2 kv heads for 8
query heads, so the raw k is SHARED across each group of 4.

That matters because F2A2's outer score is

    S[h, g] = q_h . W . k_g

If k_g is the same vector for every g in a group, the H x H score has at most
n_kv distinct columns. alpha can then express n_kv preferences, not H, and within
a group it cannot tell two heads apart at all. The mechanism is degenerate before
the first step -- and it would present as "alpha stayed at the identity, the idea
is inert", which is indistinguishable from an honest null.

Per-head query maps (item 1.4, --per-head-q) are the fix, and the reason F2A2 is
naturally a FOLLOW-ON to them rather than an alternative. install_per_head_q
applies q~_h = R_h^T q_h, so the attention score is

    (R_h^T q_h) . c_j  =  q_h . (R_h c_j)

i.e. head h's EFFECTIVE key is R_h c_j -- distinct per head even though the
latent c_j is shared. F2A2 must read THOSE, not the shared latent.

FOUR ARMS, and the middle two are the ones that make this a test rather than a
demo:

  gqa            shared keys, fixed W_O. The baseline.
  gqa+phq        per-head query maps, no mixing. Isolates 1.4 on its own, so any
                 F2A2 gain has to beat it rather than beat plain GQA.
  gqa+f2a2-naive F2A2 reading the SHARED k. PREDICTION: alpha is pinned near
                 uniform-within-group and the arm cannot beat gqa, because the
                 score has no within-group g dependence. This is the falsifiable
                 half -- if this arm DOES move, the algebra above is wrong.
  gqa+phq+f2a2   F2A2 reading the per-head effective keys R_g c. The only
                 configuration where the competition has distinct columns.

The task is the channel-retrieval one from test_f2a2.py. Note what it can and
cannot show: plain MHA already scores ~64% there, because the request sits AT the
query position so a single head can fetch the named channel and no
specialisation is required. So this is NOT a task where input-dependent head
mixing is forced to help, and a null here does not prove F2A2 useless in
general. What it CAN show, and what is decision-relevant, is whether alpha moves
off its initialisation in our key layout -- i.e. whether the mechanism is even
reachable by gradient descent here. Constructing a task where input-dependent
head mixing is provably necessary is a research question, not a smoke test, and
pretending otherwise is how the first version of this file ended up measuring
nothing.

    python experiments/test_f2a2_gqa.py --heads 8 --n-kv 2 --steps 400
"""
import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")


def make_batch(bs, n, channels, vocab, device, gen):
    ch = torch.randint(0, channels, (bs, n), generator=gen, device=device)
    val = torch.randint(0, vocab, (bs, n), generator=gen, device=device)
    want = torch.randint(0, channels, (bs,), generator=gen, device=device)
    x = ch * vocab + val
    x[:, -1] = channels * vocab + want
    y = torch.zeros(bs, dtype=torch.long, device=device)
    for b in range(bs):
        idx = (ch[b, :-1] == want[b]).nonzero()
        y[b] = val[b, idx[-1, 0]] if idx.numel() else 0
    return x, y


class GQAF2A2(nn.Module):
    """GQA (n_kv < n_heads) with optional per-head query maps and optional F2A2.

    The point of interest is `key_source`:
      "shared"   -- k_g is the group's shared key. What a naive integration does.
      "perhead"  -- k_g = R_g c, head g's effective key. Requires per_head_q.
    """

    def __init__(self, d_model, n_heads, n_kv, per_head_q=False, mix=False,
                 diag_init=4.0, key_source="shared", causal=True):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv == 0
        self.h, self.n_kv = n_heads, n_kv
        self.d = d_model // n_heads
        self.rep = n_heads // n_kv
        self.causal, self.mix, self.per_head_q = causal, mix, per_head_q
        self.key_source = key_source
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, n_kv * self.d, bias=False)   # SHARED keys
        self.v = nn.Linear(d_model, n_kv * self.d, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        if per_head_q:
            # identity at init, exactly as install_per_head_q does
            self.R = nn.Parameter(torch.eye(self.d).unsqueeze(0)
                                  .repeat(n_heads, 1, 1))
        if mix:
            self.W = nn.Parameter(torch.eye(self.d))
            self.head_scale = nn.Parameter(torch.zeros(n_heads))
            self.diag_bias = nn.Parameter(torch.full((n_heads,), float(diag_init)))
            self.register_buffer("_eye", torch.eye(n_heads), persistent=False)

    def forward(self, x, return_stats=False):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.d)
        k = self.k(x).view(B, N, self.n_kv, self.d)
        v = self.v(x).view(B, N, self.n_kv, self.d)
        if self.per_head_q:                       # q~_h = R_h^T q_h
            q = torch.einsum("bnhd,hde->bnhe", q, self.R)
        # GQA: replicate the shared k/v to every head in the group
        kr = k.repeat_interleave(self.rep, dim=2)
        vr = v.repeat_interleave(self.rep, dim=2)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2),
            is_causal=self.causal).transpose(1, 2)            # (B,N,H,d)
        alpha = None
        if self.mix:
            if self.key_source == "perhead":
                # head g's EFFECTIVE key: R_g c. Only exists with per_head_q.
                kk = torch.einsum("bngd,gde->bnge", kr, self.R)
            else:
                kk = kr                                       # shared: g-blind
            s = torch.einsum("bnhd,de,bnge->bnhg", q, self.W, kk)
            s = s / math.sqrt(self.d) * self.head_scale.exp().view(1, 1, -1, 1)
            s = s + self.diag_bias.view(1, 1, -1, 1) * self._eye
            alpha = s.softmax(-1)
            o = alpha @ o
        out = self.o(o.reshape(B, N, self.h * self.d))
        if not return_stats:
            return out
        st = {}
        if alpha is not None:
            a = alpha.detach()
            p = a.clamp_min(1e-9)
            # WITHIN-GROUP spread is the discriminating statistic. With shared
            # keys the score cannot vary across g inside a group, so every row
            # must be flat there however much the diagonal bias shifts it.
            ag = a.view(B, N, self.h, self.n_kv, self.rep)
            raw = float((ag.max(-1).values - ag.min(-1).values).mean())
            # EXCLUDE the group that contains h. diag_bias adds to entry (h,h)
            # only, so that group's spread is ~alpha_diag - off_diag whatever the
            # keys do, and it swamps everything: measured 2026-09-25, the naive
            # arm read 4.00e-01 and 99% of it was reproduced by
            # (alpha_diag - off)/n_kv from the bias alone. The first version of
            # this metric therefore could not distinguish "keys carry a signal"
            # from "there is a diagonal bias", which was the entire question.
            gi = torch.arange(self.h, device=a.device) // self.rep     # group of h
            keep = torch.ones(self.h, self.n_kv, device=a.device, dtype=torch.bool)
            keep[torch.arange(self.h, device=a.device), gi] = False
            sp = (ag.max(-1).values - ag.min(-1).values)               # (B,N,H,n_kv)
            off_group = float(sp[:, :, keep].mean()) if keep.any() else float("nan")
            st.update(alpha_diag=float(a.diagonal(dim1=-2, dim2=-1).mean()),
                      alpha_entropy=float(-(p * p.log()).sum(-1).mean()),
                      within_group_spread=raw,
                      offdiag_group_spread=off_group)
        return out, st


class Tiny(nn.Module):
    def __init__(self, attn, d, n_tok, n_out):
        super().__init__()
        self.emb = nn.Embedding(n_tok, d)
        self.pos = nn.Parameter(torch.randn(1, 256, d) * 0.02)
        self.attn = attn
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

    def forward(self, x, return_stats=False):
        h = self.emb(x) + self.pos[:, : x.shape[1]]
        out = self.attn(self.norm(h), return_stats=return_stats)
        a, st = out if return_stats else (out, {})
        return self.head((h + a)[:, -1]), st


ARMS = {
    "gqa":             dict(per_head_q=False, mix=False),
    "gqa+phq":         dict(per_head_q=True,  mix=False),
    "gqa+f2a2-naive":  dict(per_head_q=False, mix=True,  key_source="shared"),
    "gqa+phq+f2a2":    dict(per_head_q=True,  mix=True,  key_source="perhead"),
}


def run(kind, a, device):
    gen = torch.Generator(device=device).manual_seed(a.seed)
    torch.manual_seed(a.seed)
    n_tok = a.channels * a.vocab + a.channels + 1
    attn = GQAF2A2(a.d, a.heads, a.n_kv, **ARMS[kind])
    m = Tiny(attn, a.d, n_tok, a.vocab).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(a.steps):
        x, y = make_batch(a.bs, a.n, a.channels, a.vocab, device, gen)
        logits, _ = m(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    correct = tot = 0
    stats = {}
    with torch.no_grad():
        for _ in range(20):
            x, y = make_batch(a.bs, a.n, a.channels, a.vocab, device, gen)
            lg, st = m(x, return_stats=True)
            correct += (lg.argmax(-1) == y).sum().item(); tot += y.numel()
            stats = st or stats
    return correct / tot, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-kv", type=int, default=2, help="shared key heads, as MLA kept")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"channel-retrieval, GQA {a.heads} query heads / {a.n_kv} kv heads "
          f"(rep {a.heads // a.n_kv}), {a.channels} channels, {a.steps} steps, "
          f"{a.seeds} seeds, chance {1 / a.vocab:.1%}\n", flush=True)
    rows = {}
    for kind in ARMS:
        accs, ds, ws, os_ = [], [], [], []
        for s in range(a.seeds):
            a.seed = 1234 + s
            acc, st = run(kind, a, device)
            accs.append(acc)
            if st.get("alpha_diag") is not None:
                ds.append(st["alpha_diag"]); ws.append(st["within_group_spread"])
                os_.append(st["offdiag_group_spread"])
        mean = sum(accs) / len(accs)
        rows[kind] = mean
        extra = ""
        if ds:
            extra = (f"\n      alpha diag {sum(ds)/len(ds):.3f}  "
                     f"within-group spread {sum(ws)/len(ws):.2e} (diag-dominated, "
                     f"ignore)\n      KEY-DRIVEN spread {sum(os_)/len(os_):.2e}  "
                     f"<- groups excluding h, so diag_bias contributes nothing. "
                     f"~0 => shared keys give the competition nothing to see")
        print(f"  {kind:<16} acc {mean:.1%}  (spread {max(accs)-min(accs):.1%}, "
              f"{[f'{x:.0%}' for x in accs]}){extra}", flush=True)
    print()
    print(f"  phq over gqa          {rows['gqa+phq'] - rows['gqa']:+.1%}"
          f"   (item 1.4 alone)")
    print(f"  f2a2-naive over gqa   {rows['gqa+f2a2-naive'] - rows['gqa']:+.1%}"
          f"   (predicted ~0: shared keys give no within-group signal)")
    print(f"  phq+f2a2 over phq     {rows['gqa+phq+f2a2'] - rows['gqa+phq']:+.1%}"
          f"   <- the only number that would justify F2A2")


if __name__ == "__main__":
    main()
