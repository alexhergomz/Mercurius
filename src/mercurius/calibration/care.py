"""CARE-style MLA conversion: whitened factorization, measured against plain SVD.

The comparison that matters. Plain SVD minimizes ||W - W_hat||, which is the
wrong objective -- what the model actually cares about is ||XW - XW_hat||.
CARE / SVD-LLM whiten by the calibration covariance first so the truncation
optimizes activation error. CARE reports ~19.50 ppl against plain Palu(SVD)'s
~45.40 at the same rank on Llama-3.1-8B.

Calibration is cheap: CARE uses 256 samples of length 32 and says performance
saturates past 512 samples. We use more tokens than that because it costs
seconds here and the covariance is only 1024x1024.

Runs both factorizations at matched ranks so the delta is attributable to the
whitening and nothing else.
"""
import sys, json, argparse, torch
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora, freeze_base
from mercurius.surgery.transmla import convert_to_mla, choose_rank
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.paths import CKPT_DIR, FINEWEB, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
CALIB = str(FINEWEB)
ADAPTERS = str(CKPT_DIR / 'adapters-combined.pt')
LORA_RULES = [
    ("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
    ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),
    ("linear_attn.out_proj", 16), ("linear_attn.in_proj_qkv", 16),
    ("mlp.gate_proj", 16), ("mlp.up_proj", 16), ("mlp.down_proj", 16),
    ("lm_head", 0), ("embed_tokens", 0),
]
LENGTHS = [2048, 8192, 32768]


def build():
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    for l in get_trunk(m).layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")
    inject_lora(m, LORA_RULES, verbose=False)
    freeze_base(m)
    sd = torch.load(ADAPTERS, map_location="cpu")
    m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    return m.eval()


@torch.no_grad()
def collect_covariances(model, ids, n_samples=256, seq=512, seed=0):
    """X^T X of the hidden states entering each attention layer.

    Accumulated in float64: these are sums over ~130k vectors and the matrix is
    then Cholesky-factorized, so precision here is worth the negligible cost.
    """
    trunk = get_trunk(model)
    attn_idx = [i for i, l in enumerate(trunk.layers) if hasattr(l, "self_attn")]
    d = trunk.layers[0].input_layernorm.weight.shape[0]
    covs = {i: torch.zeros(d, d, dtype=torch.float64, device="cuda") for i in attn_idx}
    counts = {i: 0 for i in attn_idx}

    hooks = []
    def mk(i):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).double()
            covs[i] += x.T @ x
            counts[i] += x.shape[0]
        return hook
    for i in attn_idx:
        hooks.append(trunk.layers[i].self_attn.k_proj.register_forward_hook(mk(i)))

    g = torch.Generator().manual_seed(seed)
    for s in range(n_samples):
        off = int(torch.randint(0, len(ids) - seq - 1, (1,), generator=g))
        model(input_ids=ids[off:off + seq].unsqueeze(0).cuda(), logits_to_keep=1)
        if (s + 1) % 64 == 0:
            print(f"    calibrated {s+1}/{n_samples}", flush=True)
        torch.cuda.empty_cache()
    for h in hooks:
        h.remove()
    for i in attn_idx:
        covs[i] /= max(counts[i], 1)
    print(f"    covariance from {counts[attn_idx[0]]:,} token vectors", flush=True)
    return {i: c.float() for i, c in covs.items()}


def measure(m, ids, needle, tag):
    row = {"tag": tag}
    for n in LENGTHS:
        row[f"ppl{n}"] = perplexity(m, ids, n)
        torch.cuda.empty_cache()
    _, _, row["retr"] = retrieval(m, ids, needle, 4096)
    torch.cuda.empty_cache()
    print(f"  {tag:<34}" + "".join(f"{row[f'ppl{n}']:>10.3f}" for n in LENGTHS)
          + f"{row['retr']:>10.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--calib-seq", type=int, default=512)
    ap.add_argument("--out", default="logs/care.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    calib = tok(open(CALIB).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    hdr = ("  " + "config".ljust(34) + "".join(f"{n:>10}" for n in LENGTHS)
           + f"{'retr@4k':>10}")

    print("=== baseline (NoPE trained, no MLA) ===")
    print(hdr, flush=True)
    m = build()
    rows = [measure(m, ids, needle, "baseline")]
    base = rows[0]

    print(f"\n=== calibration: {a.calib_samples} x {a.calib_seq} tokens ===", flush=True)
    covs = collect_covariances(m, calib, a.calib_samples, a.calib_seq)
    del m
    torch.cuda.empty_cache()

    print("\n=== plain SVD vs CARE whitening, at MATCHED ranks ===")
    print(hdr, flush=True)
    for d_c in (729, 512, 384, 256):
        for use_cov, label in ((False, "plain SVD"), (True, "CARE whitened")):
            m = build()
            convert_to_mla(m, d_c=d_c, verbose=False,
                           covs=covs if use_cov else None)
            r = measure(m, ids, needle, f"d_c={d_c:<4} {label}")
            r.update(d_c=d_c, whitened=use_cov)
            rows.append(r)
            del m
            torch.cuda.empty_cache()

    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")

    print("\n=== whitening gain, at matched rank (vs uncompressed baseline) ===")
    print(f"  {'d_c':<8}{'KV':>7}{'plain SVD':>14}{'CARE':>14}{'gain':>10}")
    for d_c in (729, 512, 384, 256):
        p = next(r for r in rows[1:] if r.get("d_c") == d_c and not r["whitened"])
        c = next(r for r in rows[1:] if r.get("d_c") == d_c and r["whitened"])
        dp = (p["ppl2048"] - base["ppl2048"]) / base["ppl2048"] * 100
        dc = (c["ppl2048"] - base["ppl2048"]) / base["ppl2048"] * 100
        print(f"  {d_c:<8}{1024/d_c:>6.2f}x{dp:>13.2f}%{dc:>13.2f}%{dp-dc:>9.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
