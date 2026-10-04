"""Artifacts for giving the student the TEACHER's output head.

    logits = W_t (P h_s + b),   P: 2560 -> 2048,   W_t frozen (248320 x 2048)

WHY THE TEACHER'S HEAD AND NOT A MAP BACK INTO THE STUDENT'S. Both were measured
on 321 real teacher hidden states. Reconstructing the teacher's logits through
the STUDENT's head loses 18% of top-1 predictions (KL 0.433) even with the most
favourable fit -- against KL 0.0017 for putting 5% random noise on h. Decoding
through the teacher's own head has zero reconstruction error by construction.
D13 is the precedent for why the "95.2% of energy explained" figure did not
survive: lm_head amplifies small h differences, so energy badly overstates
fidelity.

WHY P IS FITTED TO PRESERVE THE STUDENT, NOT TO MATCH THE TEACHER. Measured on
60,000 paired states, fitted on calib_mix's code-heavy 80% and evaluated on its
prose tail:

    init for P                        disruption vs student   agree w/ teacher
    fit P h_s ~= h_t                  KL 0.7401  top-1 74.0%        69.7%
    fit W_t P ~= W_s, uniform tokens  KL 1.0461  top-1 66.3%        61.7%
    fit W_t P ~= W_s, usage-weighted  KL 0.0663  top-1 94.2%        82.8%
    (student today)                                                 83.9%

Fitting P to the training target destroys the model it starts from. Fitting it to
preserve the student's own logits keeps 94.2% of top-1 and leaves teacher
agreement essentially where it was -- which is what the roadmap means by a
function-preserving initialization. Note the rank argument (2048 < 2560) does NOT
make this lossy in practice; UNIFORM token weighting does, catastrophically.

THE METRIC. Hidden-state MSE equals a logit-space quantity only under the right
metric; plain L2 misweights directions by ~900x. Two are written:

    G  = W^T (I - 11^T/V) W            centred logit MSE, needs no softmax
    F  = E[ W^T diag(p) W - mu mu^T ]  the Fisher: 1/2 d^T F d == KL to 2nd order

F is the principled one (it puts the loss in NATS, so ce_beta keeps its meaning),
and is accumulated over real teacher states. Verified against true KL at the
actual starting delta: 13.1% high at s=1.0, 5.4% at s=0.5, 0.6% at s=0.05 -- the
error SHRINKS as training converges, which is the opposite of D12's top-k
artefact that was a floor exactly where resolution was needed.

    python scripts/make_head_swap.py --out cache/head_swap_35b.pt
"""
import argparse, collections, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from mercurius.paths import ROOT, STAGE_AB
from safetensors import safe_open

HEAD = "model.language_model.embed_tokens.weight"


def student_head(d):
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(d, fn), framework="pt") as f:
                if HEAD in f.keys():
                    return f.get_tensor(HEAD).float()
    raise SystemExit("student head not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-head", default=str(ROOT / "cache/teacher35b_head.npy"))
    ap.add_argument("--teacher-cache", default=str(ROOT / "cache/h_teacher_calib.npz"))
    ap.add_argument("--fisher-positions", type=int, default=4096,
                    help="positions to accumulate the Fisher over; each costs one "
                         "full-vocabulary softmax, so this is the only expensive step")
    ap.add_argument("--out", default=str(ROOT / "cache/head_swap_35b.pt"))
    a = ap.parse_args()

    Wt = torch.from_numpy(np.load(a.teacher_head))          # (V, d_t)
    Ws = student_head(str(STAGE_AB))                        # (V, d_s)
    V, dt = Wt.shape
    ds = Ws.shape[1]
    print(f"teacher head {tuple(Wt.shape)}   student head {tuple(Ws.shape)}")

    t = np.load(a.teacher_cache)
    Ht = torch.from_numpy(t["h_int8"].astype(np.float32) * t["scale"])
    ids = t["ids"]
    # usage weights from the corpus the states came from
    p = torch.zeros(V, dtype=torch.float64)
    for k, v in collections.Counter(ids.tolist()).items():
        if k < V:
            p[k] = v
    p = p / p.sum()
    p = 0.999999 * p + 1e-6 / V

    Wtc = (Wt - Wt.mean(0, keepdim=True))
    Wsc = (Ws - Ws.mean(0, keepdim=True))
    sw = p.sqrt().unsqueeze(1).float()
    A = (Wtc * sw).T @ (Wtc * sw)
    A = A.double()
    M = ((Wtc * sw).T @ (Wsc * sw)).double()
    P = torch.linalg.solve(A + 1e-6 * torch.diag(A).mean() * torch.eye(dt, dtype=torch.float64), M)
    print(f"P {tuple(P.shape)}  (usage-weighted, preserves the student's own logits)")

    # centred-logit Gram, no softmax needed
    G = (Wtc.T @ Wtc).double() / V

    # The GLOBAL Fisher, exactly, without touching a per-position d x d matrix.
    #
    #   F = E_h[ W^T diag(p_h) W - mu_h mu_h^T ]
    #     = W^T diag( E_h[p_h] ) W  -  E_h[ mu_h mu_h^T ]
    #
    # The first term needs only the AVERAGE predicted distribution p_bar, a
    # single V-vector, so it costs ONE (d,V)@(V,d) product rather than one per
    # position. Written the naive way it is ~1 TFLOP per position and needs a
    # (V, batch, d) intermediate that does not fit.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Wd = Wtc.to(dev)
    n = min(a.fisher_positions, Ht.shape[0])
    sel = torch.linspace(0, Ht.shape[0] - 1, n).long()
    pbar = torch.zeros(V, dtype=torch.float64, device=dev)
    MM = torch.zeros(dt, dt, dtype=torch.float64, device=dev)
    bs = 64
    for i in range(0, n, bs):
        h = Ht[sel[i:i + bs]].to(dev)
        pr = (h @ Wd.T).softmax(-1)                     # (b, V)
        pbar += pr.sum(0).double()
        mu = (pr @ Wd).double()                         # (b, d_t)
        MM += mu.T @ mu
        if (i + bs) % 1024 == 0:
            print(f"  fisher {min(i+bs, n)}/{n}", flush=True)
    pbar /= n
    MM /= n
    F = ((Wd.double() * pbar.unsqueeze(1)).T @ Wd.double() - MM).cpu()
    top = pbar.sort(descending=True).values
    print(f"Fisher {tuple(F.shape)} over {n} positions; mean teacher distribution "
          f"puts {100*float(top[:64].sum()):.1f}% of mass in its top-64, "
          f"{100*float(top[:1024].sum()):.1f}% in top-1024")

    torch.save({"P": P.float(), "W_t": Wt, "G": G.float(), "F": F.float(),
                "d_s": ds, "d_t": dt, "V": V,
                "note": "logits = W_t (P h_s); P preserves the student's logits "
                        "(usage-weighted); loss metric F (nats) or G (logit MSE)"},
               a.out)
    print(f"-> {a.out} ({os.path.getsize(a.out)/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
