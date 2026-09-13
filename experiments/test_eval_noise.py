"""How reproducible is ppl@8192 on this machine?

Several conclusions in this project rest on differences of 0.5-1.7% in
perplexity between checkpoints. None of them mean anything without a noise
floor, and the floor was never measured.

Two sources are separated here:

  within-model   the same weights evaluated repeatedly. bf16 accumulation and
                 FlashAttention's reduction order are not bitwise deterministic
                 across launches, so this is not guaranteed to be zero.

  across-reload  the same checkpoint loaded fresh each time. Adds any
                 nondeterminism in construction, conversion and weight loading
                 on top of the above.

If within-model variance is ~0, a 1% gap between two checkpoints is real. If it
is ~1%, then the step-100-versus-step-150 "drift" seen in val-full, val-ctrl and
ls8k is an artifact and the claims built on it have to go.
"""
import argparse
import sys
import torch
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.surgery.transmla import convert_to_mla
from mercurius.adapters.lora import inject_lora, freeze_base
from mercurius.eval.characterize import perplexity
from mercurius.recovery.train import CKPT, EVAL_DATA, LORA_RULES
from mercurius.paths import CACHE_DIR


def build(adapters, dc, covs_path):
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    for l in m.model.language_model.layers if hasattr(m.model, "language_model") \
            else m.model.layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")
    inject_lora(m, LORA_RULES, verbose=False)
    freeze_base(m)
    if dc:
        covs = {int(k): v.cuda().float()
                for k, v in torch.load(covs_path, map_location="cpu").items()}
        convert_to_mla(m, d_c=dc, covs=covs, verbose=False)
    sd = torch.load(adapters, map_location="cpu")
    m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="ckpt/adapters-val-full.pt")
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--covs", default=str(CACHE_DIR / 'kv_covs.pt'))
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--reloads", type=int, default=3)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(EVAL_DATA).read(), return_tensors="pt").input_ids[0]

    def stats(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
        return m, sd, (max(xs) - min(xs)) / m * 100

    print(f"eval: ppl@{a.n} on {a.adapters.split('/')[-1]}\n")

    model = build(a.adapters, a.dc, a.covs)
    within = [perplexity(model, ids, a.n) for _ in range(a.repeats)]
    for i, v in enumerate(within):
        print(f"  within-model  run {i+1}  {v:.6f}")
    m, sd, spread = stats(within)
    print(f"  -> mean {m:.4f}  sd {sd:.2e}  spread {spread:.4f}%\n")
    del model
    torch.cuda.empty_cache()

    across = []
    for i in range(a.reloads):
        mdl = build(a.adapters, a.dc, a.covs)
        v = perplexity(mdl, ids, a.n)
        across.append(v)
        print(f"  across-reload run {i+1}  {v:.6f}")
        del mdl
        torch.cuda.empty_cache()
    m2, sd2, spread2 = stats(across)
    print(f"  -> mean {m2:.4f}  sd {sd2:.2e}  spread {spread2:.4f}%\n")

    print("  the differences these numbers have to adjudicate:")
    print("    val-full  step100 17.292 vs step150 17.432   0.81%")
    print("    val-ctrl  step100 17.130 vs step150 17.424   1.72%")
    print("    ls8k      step100 17.302 vs step150 17.463   0.93%")
    worst = max(spread, spread2)
    print(f"\n  measured spread {worst:.4f}% -> "
          f"{'those gaps are REAL' if worst < 0.4 else 'those gaps are NOT separable from noise'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
