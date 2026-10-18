"""Fit the teacher->student head alignment and the metric that makes
hidden-state MSE equal expected logit MSE.

WHY THIS IS POSSIBLE AT ALL. The two models share a vocabulary, so their output
heads are row-indexed by the SAME symbols: 248,320 PAIRED observations. That is
an anchor set, and it makes the alignment a closed-form weighted least squares
rather than something to learn. (Same idea as anchor-based "relative
representations"; the space-mismatch problem itself is DSKD's "space
discrepancy", arXiv:2406.17328.)

WHY WEIGHTING BY TOKEN USAGE IS NOT OPTIONAL. An unweighted fit treats all
248,320 rows equally. Only 18,008 of them ever occur in our corpus and the top
1,000 carry 76.3% of the mass, so the unweighted fit is dominated by tokens the
model never emits. Measured, same matrices, only the weights differ:

    teacher head expressible through the student's   uniform 28.2%  usage 89.9%
    student head expressible through the teacher's   uniform 47.9%  usage 97.3%

The uniform numbers say the design is hopeless; the usage-weighted numbers say
it works. A rank argument (2560 < 5120) sets a ceiling under uniform weights
that the usage-weighted fit simply exceeds -- the frequent-token rows of the two
heads largely co-span, which unweighted Frobenius cannot see.

WHAT IS PRODUCED
    B  (d_s x d_t)  target map: the student's target is `B @ h_t`, a vector in
                    the STUDENT's own space. Its decoding through the student's
                    UNCHANGED head reproduces the teacher's logits to ~90% of
                    their usage-weighted energy. Nothing is added at inference.
    G  (d_s x d_s)  metric: sum_v p(v) w~_s[v] w~_s[v]^T, so

                        (h_s - B h_t)^T G (h_s - B h_t)

                    is the EXPECTED squared logit error under the token
                    distribution. Plain L2 on h would misweight directions by
                    ~900x (measured on the 4B head), so the metric is not
                    cosmetic.

Both heads are centred over the vocabulary first: softmax ignores a constant
added to every entry, so an uncentred fit pays for a direction that changes no
probability.

    python scripts/make_head_align.py --episodes 300
"""
import argparse
import collections
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "src")
from mercurius.paths import DATA_DIR, ROOT, STAGE_AB, TEACHER_MODEL

HEAD_NAMES = ["lm_head.weight", "model.embed_tokens.weight",
              "model.language_model.embed_tokens.weight"]


def load_head(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        m = json.load(open(idx))["weight_map"]
        for n in HEAD_NAMES:
            if n in m:
                with safe_open(os.path.join(d, m[n]), framework="pt") as f:
                    return f.get_tensor(n), n
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(d, fn), framework="pt") as f:
                for n in HEAD_NAMES:
                    if n in f.keys():
                        return f.get_tensor(n), n
    raise SystemExit(f"no output head found under {d}")


def token_freq(tok, path, n_episodes, V):
    cnt = collections.Counter()
    seen = 0
    with open(path) as fh:
        for line in fh:
            ep = json.loads(line)
            text = tok.apply_chat_template(ep["messages"], tools=ep.get("tools"),
                                           tokenize=False)
            cnt.update(tok(text, add_special_tokens=False).input_ids)
            seen += 1
            if seen >= n_episodes:
                break
    p = torch.zeros(V, dtype=torch.float64)
    for k, v in cnt.items():
        if k < V:
            p[k] = v
    return p / p.sum(), seen, int(sum(cnt.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(STAGE_AB))
    ap.add_argument("--teacher", default=str(TEACHER_MODEL))
    ap.add_argument("--corpus", default=str(DATA_DIR / "episodes/pilot_mix.jsonl"))
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--floor", type=float, default=1e-6, metavar="F",
                    help="uniform mass mixed into p(v) so unseen tokens are not "
                         "weighted at exactly zero -- a token absent from 300 "
                         "episodes is rare, not impossible, and a hard zero lets "
                         "the fit place arbitrary error on it")
    ap.add_argument("--out", default=str(ROOT / "cache/head_align_27b.pt"))
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.student)
    Ws, ns = load_head(a.student)
    Wt, nt = load_head(a.teacher)
    V, ds = Ws.shape
    dt = Wt.shape[1]
    if Wt.shape[0] != V:
        raise SystemExit(f"vocabularies differ: student {V}, teacher {Wt.shape[0]}")
    print(f"student head {ns} {tuple(Ws.shape)}")
    print(f"teacher head {nt} {tuple(Wt.shape)}")

    p, n_ep, n_tok = token_freq(tok, a.corpus, a.episodes, V)
    p = (1 - a.floor) * p + a.floor / V
    print(f"token usage from {n_ep} episodes / {n_tok:,} tokens: "
          f"{int((p > a.floor / V).sum()):,} distinct, top-1000 carry "
          f"{100 * p.sort(descending=True).values[:1000].sum():.1f}% of mass")

    Ws = Ws.float() - Ws.float().mean(0, keepdim=True)
    Wt = Wt.float() - Wt.float().mean(0, keepdim=True)
    sw = p.sqrt().unsqueeze(1).float()
    A, Bm = Ws * sw, Wt * sw                       # usage-weighted rows
    G = (A.T @ A).double()                         # d_s x d_s   -- the metric
    M = (A.T @ Bm).double()                        # d_s x d_t
    ridge = 1e-6 * torch.diag(G).mean()
    B = torch.linalg.solve(G + ridge * torch.eye(ds, dtype=torch.float64), M)

    expl = float((B * M).sum())
    tot = float((Bm.double() * Bm.double()).sum())
    print(f"\nusage-weighted teacher logit energy reproduced through the "
          f"student's own head: {100 * expl / tot:.1f}%")
    ev = torch.linalg.eigvalsh(G).clamp_min(0)
    print(f"metric G: eigenvalues {ev.min():.2e} .. {ev.max():.2e} "
          f"(plain L2 would misweight by {ev.max() / ev.clamp_min(ev.max() * 1e-6).min():.0f}x)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"B": B.float(), "G": G.float(), "explained": expl / tot,
                "student_head": ns, "teacher_head": nt, "d_s": ds, "d_t": dt,
                "n_episodes": n_ep, "n_tokens": n_tok, "floor": a.floor,
                "note": "target = B @ h_t in student space; "
                        "loss = (h_s - B h_t)^T G (h_s - B h_t) == expected logit MSE"},
               a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
