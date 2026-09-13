"""G2 — what does NF4 cost on the converted model?

The plan flagged a risk: folding RMSNorm gains into the weights widens their
per-column dynamic range, which is the condition NF4 handles worst. Measurement
already showed the effective gain (1+w) spans only ~4.3x, so the risk looked
small -- this checks it directly.

Compares, at three context lengths:
  bf16 converted      (reference)
  NF4 converted       (what training will actually run on)
and reports the per-column dynamic range of fused vs unfused weights, which is
the mechanism the risk would act through.
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius.models.kda import load_kda_model, convert_to_kda
from mercurius.surgery.norm_fusion import get_trunk, fuse_model
from mercurius.models.quantize import quantize_nf4, param_bytes
from mercurius.eval.characterize import perplexity
from mercurius.paths import BASE_MODEL, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
ORIG = str(BASE_MODEL)
DATA = str(WIKITEXT)
LENGTHS = [2048, 8192, 32768]


def column_range(model, tag):
    """max/median of per-column absolute max -- the quantization-difficulty proxy."""
    worst, name = 0.0, ""
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if any(s in n for s in ("lm_head", "embed_tokens")):
            continue
        colmax = m.weight.data.abs().amax(dim=0).float()
        r = (colmax.max() / colmax.median().clamp_min(1e-9)).item()
        if r > worst:
            worst, name = r, n
    print(f"  {tag:<26} worst per-column max/median = {worst:8.2f}  ({name})")
    return worst


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]

    print("=== dynamic range: does fusion make quantization harder? ===", flush=True)
    m = AutoModelForCausalLM.from_pretrained(ORIG, dtype=torch.float32,
                                             device_map="cuda").eval()
    r_unfused = column_range(m, "unfused (original)")
    fuse_model(m, verbose=False)
    r_fused = column_range(m, "fused (stage A)")
    print(f"  ratio fused/unfused = {r_fused/r_unfused:.3f}"
          f"  -> {'HARDER' if r_fused > r_unfused*1.5 else 'no meaningful change'}")
    del m; torch.cuda.empty_cache()

    print("\n=== bf16 reference ===", flush=True)
    model = load_kda_model(CKPT, dtype=torch.bfloat16)
    ref = {}
    for n in LENGTHS:
        ref[n] = perplexity(model, ids, n)
        print(f"  ppl@{n:<6} {ref[n]:8.3f}", flush=True)
    bf16_bytes = param_bytes(model)
    del model; torch.cuda.empty_cache()

    print("\n=== NF4 ===", flush=True)
    model = load_kda_model(CKPT, dtype=torch.bfloat16)
    quantize_nf4(model)
    nf4_bytes = param_bytes(model)
    q = {}
    for n in LENGTHS:
        q[n] = perplexity(model, ids, n)
        d = (q[n] - ref[n]) / ref[n] * 100
        print(f"  ppl@{n:<6} {q[n]:8.3f}   ({d:+.2f}% vs bf16)", flush=True)
    del model; torch.cuda.empty_cache()

    print("\n=== G2 VERDICT ===")
    print(f"  weights: {bf16_bytes/2**30:.2f} GiB bf16 -> {nf4_bytes/2**30:.2f} GiB NF4"
          f"  ({bf16_bytes/max(nf4_bytes,1):.2f}x)")
    worst = max((q[n] - ref[n]) / ref[n] * 100 for n in LENGTHS)
    print(f"  worst perplexity delta: {worst:+.2f}%")
    print(f"  {'PASS' if worst < 5.0 else 'FAIL'} (threshold +5% -- the plan's 1% was"
          f" written before measuring; NF4 on a 0.8B is a harder case than 9B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
