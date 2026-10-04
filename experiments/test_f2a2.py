"""Smoke test for F2A2: does letting heads compete buy anything?

The task is built so that head specialisation is the ONLY way to do well.
Each sequence carries several interleaved "channels"; a query token names the
channel it wants, and the answer is the most recent token of that channel. With
H heads and C channels, a head can specialise to a channel, but a FIXED head
mixture (standard MHA's W_O) must average over all of them at every position.
An input-dependent convex mixture can put its mass on the head that happens to
hold the requested channel -- which is the mute limit F2A2 is meant to break.

This is deliberately a task F2A2 should win. A smoke test answers "is the
mechanism alive and does it do what it claims", not "is it better in general".
Three things are checked, and the third is the one that matters:

  1. does it learn the task at all
  2. does it beat a parameter-matched MHA control
  3. do heads actually BORROW? alpha is an H x H distribution; if it collapses to
     the identity every head just keeps its own pattern, the layer is MHA with
     H x the FLOPs, and any difference is noise or the extra 264 parameters.
     alpha's diagonal mass is therefore reported, not assumed.

ARMS:
  mha        plain MHA, the reference
  no-mix     F2A2 with alpha pinned to the identity == bit-for-bit MHA. Proves
             the control is honest (must match `mha` EXACTLY) and isolates the
             mechanism from the +264 parameters.
  f2a2-d0    naive init: alpha starts uniform, so the layer starts by giving
             every head the average pattern -- further from MHA than it looks.
  f2a2-d4    alpha starts near the identity, so the layer starts AT MHA and the
             task has to pay for borrowing. This is the honest test of whether
             head-borrowing earns its keep.

If d4 ends up with alpha_diag still ~1, borrowing was never worth paying for and
the mechanism is inert -- which is the result, not a bug to tune away.

    python experiments/test_f2a2.py --heads 8 --channels 8 --steps 400
"""
import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from mercurius.models.f2a2 import F2A2Attention, MHAReference


def make_batch(bs, n, channels, vocab, device, gen):
    """Interleaved channels; the last token asks for one and the label is that
    channel's most recent value."""
    ch = torch.randint(0, channels, (bs, n), generator=gen, device=device)
    val = torch.randint(0, vocab, (bs, n), generator=gen, device=device)
    want = torch.randint(0, channels, (bs,), generator=gen, device=device)
    # token id packs (channel, value); the final position packs the request
    x = ch * vocab + val
    x[:, -1] = channels * vocab + want
    y = torch.zeros(bs, dtype=torch.long, device=device)
    for b in range(bs):
        idx = (ch[b, :-1] == want[b]).nonzero()
        y[b] = val[b, idx[-1, 0]] if idx.numel() else 0
    return x, y


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


def run(kind, a, device):
    gen = torch.Generator(device=device).manual_seed(a.seed)
    torch.manual_seed(a.seed)
    n_tok = a.channels * a.vocab + a.channels + 1
    if kind == "mha":
        attn = MHAReference(a.d, a.heads)
    else:
        attn = F2A2Attention(a.d, a.heads, no_mix=(kind == "no-mix"),
                             diag_init=float(kind.split("-d")[1]) if "-d" in kind else 0.0)
    m = Tiny(attn, a.d, n_tok, a.vocab).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    stats = {}
    for step in range(a.steps):
        x, y = make_batch(a.bs, a.n, a.channels, a.vocab, device, gen)
        logits, _ = m(x, return_stats=False), None
        loss = F.cross_entropy(logits[0] if isinstance(logits, tuple) else logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval on fresh data
    m.eval()
    correct = tot = 0
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
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"channel-retrieval: {a.channels} channels, {a.heads} heads, "
          f"len {a.n}, {a.steps} steps, {a.seeds} seeds, chance = "
          f"{1.0 / a.vocab:.1%}\n")
    rows = {}
    for kind in ("mha", "no-mix", "f2a2-d0", "f2a2-d4"):
        accs, diags, ents = [], [], []
        for s in range(a.seeds):
            a.seed = 1234 + s
            acc, st = run(kind, a, device)
            accs.append(acc)
            if st.get("alpha_diag") is not None:
                diags.append(st["alpha_diag"]); ents.append(st["alpha_entropy"])
        mean = sum(accs) / len(accs)
        e = (f"\n           alpha: diag {sum(diags)/len(diags):.3f} (1.0 = inert, "
             f"{1.0/a.heads:.3f} = uniform), entropy {sum(ents)/len(ents):.3f} "
             f"/ {math.log(a.heads):.3f}") if diags else ""
        print(f"  {kind:<9} acc {mean:.1%}  (spread {max(accs)-min(accs):.1%}, "
              f"seeds {[f'{x:.0%}' for x in accs]}){e}")
        rows[kind] = mean
    print()
    for k in ("f2a2-d0", "f2a2-d4"):
        print(f"  {k} - no-mix  {rows[k] - rows['no-mix']:+.1%}   "
              f"(same module, alpha pinned to identity -- the honest control)")
    # Compare the MODULES, not two post-training accuracies. no_mix uses SDPA
    # where MHAReference uses a manual matmul: mathematically identical, but the
    # kernels differ in reduction order, and that rounding compounds over 400
    # steps into an accuracy gap far above any sane epsilon. Asserting on
    # accuracy therefore reported "control is NOT honest" for a control that is
    # exact -- measured max|diff| 2.8e-16 in double precision, i.e. machine eps.
    import torch as _t
    _f = F2A2Attention(a.d, a.heads, no_mix=True).double()
    _m = MHAReference(a.d, a.heads).double()
    for _s, _d in ((_f.q, _m.q), (_f.k, _m.k), (_f.v, _m.v), (_f.o, _m.o)):
        _d.weight.data.copy_(_s.weight.data)
    _x = _t.randn(4, 32, a.d, dtype=_t.double)
    with _t.no_grad():
        _err = (_f(_x) - _m(_x)).abs().max().item()
    if _err > 1e-10:
        print(f"  WARNING: no-mix is not MHA: max|diff| {_err:.3e}. Real bug.")
    else:
        print(f"  control check: no-mix IS MHA, max|diff| {_err:.1e} (machine eps).")
        print(f"  (accuracy differs by {abs(rows['no-mix']-rows['mha']):.1%} from "
              f"kernel rounding compounded over training, not from wiring.)")


if __name__ == "__main__":
    main()
