"""Stage D runner: apply TransMLA latent KV and measure what it costs.

Same discipline as every other stage here:

  1. EXACTNESS FIRST. At full rank the SVD factorization reproduces the original
     mapping exactly, so converting at d_c = full_rank must leave the model
     unchanged. If it does not, the factorization or the k/v shim is wrong and
     nothing below it is worth measuring.
  2. Then sweep the lossy ranks and report the damage against controls.

Runs on the NoPE-TRAINED model, since stage C is what unblocked D.
"""
import sys, json, argparse, torch
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora, freeze_base
from mercurius.surgery.transmla import convert_to_mla, choose_rank, merged_weight
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.paths import CKPT_DIR, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
ADAPTERS = str(CKPT_DIR / 'adapters-combined.pt')
LORA_RULES = [
    ("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
    ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),
    ("linear_attn.out_proj", 16), ("linear_attn.in_proj_qkv", 16),
    ("mlp.gate_proj", 16), ("mlp.up_proj", 16), ("mlp.down_proj", 16),
    ("lm_head", 0), ("embed_tokens", 0),
]
LENGTHS = [2048, 8192, 32768]


def build(adapters=True):
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    for l in get_trunk(m).layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)  # match the run
    install_rope_dial(m, 0, "global")          # NoPE
    if adapters:
        inject_lora(m, LORA_RULES, verbose=False)
        freeze_base(m)
        sd = torch.load(ADAPTERS, map_location="cpu")
        m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    return m.eval()


def measure(m, ids, needle, tag):
    row = {"tag": tag}
    for n in LENGTHS:
        row[f"ppl{n}"] = perplexity(m, ids, n)
        torch.cuda.empty_cache()
    _, _, row["retr"] = retrieval(m, ids, needle, 4096)
    torch.cuda.empty_cache()
    print(f"  {tag:<30}" + "".join(f"{row[f'ppl{n}']:>10.3f}" for n in LENGTHS)
          + f"{row['retr']:>10.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/stage_d.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    print("=== spectral structure of [W_k; W_v] per attention layer ===", flush=True)
    m = build()
    for i, l in enumerate(get_trunk(m).layers):
        sa = getattr(l, "self_attn", None)
        if sa is None:
            continue
        for e in (0.99, 0.95, 0.90):
            r, S = choose_rank(sa.k_proj, sa.v_proj, energy=e)
            print(f"  layer {i:>2}  energy {e:.2f} -> d_c {r:>4} / {S.numel()}"
                  + ("" if e != 0.90 else ""), flush=True)
        break   # spectra are near-identical across layers; one is enough here
    for i, l in enumerate(get_trunk(m).layers):
        sa = getattr(l, "self_attn", None)
        if sa is None:
            continue
        r99, S = choose_rank(sa.k_proj, sa.v_proj, energy=0.99)
        r95, _ = choose_rank(sa.k_proj, sa.v_proj, energy=0.95)
        print(f"  layer {i:>2}: full {S.numel():>4}  d_c@0.99 {r99:>4}  "
              f"d_c@0.95 {r95:>4}", flush=True)

    hdr = "  " + "config".ljust(30) + "".join(f"{n:>10}" for n in LENGTHS) + f"{'retr@4k':>10}"
    print("\n=== baseline (NoPE trained, no MLA) ===")
    print(hdr, flush=True)
    rows = [measure(m, ids, needle, "baseline")]
    base = rows[0]
    del m; torch.cuda.empty_cache()

    # ---- 1. exactness at full rank ----
    print("\n=== exactness check: d_c = full rank should be a no-op ===")
    print(hdr, flush=True)
    m = build()
    # full rank of the stacked [W_k; W_v]; k_proj may be LoRA-wrapped
    _wk, _ = merged_weight(get_trunk(m).layers[3].self_attn.k_proj)
    full = _wk.shape[0] * 2
    convert_to_mla(m, d_c=full, verbose=False)
    r = measure(m, ids, needle, f"full rank d_c={full}")
    rows.append(r)
    dev = abs(r["ppl2048"] - base["ppl2048"]) / base["ppl2048"] * 100
    print(f"  -> deviation from baseline: {dev:.4f}%  "
          f"{'EXACT (factorization correct)' if dev < 0.5 else 'BUG in factorization or shim'}")
    del m; torch.cuda.empty_cache()

    # ---- 2. lossy ranks ----
    print("\n=== lossy compression ===")
    print(hdr, flush=True)
    for label, kw in (("adaptive energy 0.99", dict(d_c=None, energy=0.99)),
                      ("adaptive energy 0.95", dict(d_c=None, energy=0.95)),
                      ("fixed d_c=512", dict(d_c=512)),
                      ("fixed d_c=256", dict(d_c=256))):
        m = build()
        info = convert_to_mla(m, verbose=False, **kw)
        saving = sum(fr for _, _, fr, _ in info) / sum(rr for _, rr, _, _ in info)
        rr = measure(m, ids, needle, f"{label} ({saving:.2f}x KV)")
        rr["kv_saving"] = saving
        rr["ranks"] = [x[1] for x in info]
        rows.append(rr)
        del m; torch.cuda.empty_cache()

    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")

    print("\n=== cost of compression, vs the uncompressed NoPE-trained model ===")
    for r in rows[2:]:
        d2 = (r["ppl2048"] - base["ppl2048"]) / base["ppl2048"] * 100
        d32 = (r["ppl32768"] - base["ppl32768"]) / base["ppl32768"] * 100
        dr = r["retr"] - base["retr"]
        print(f"  {r['tag']:<34} ppl@2048 {d2:+6.2f}%  ppl@32768 {d32:+6.2f}%  "
              f"retrieval {dr:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
