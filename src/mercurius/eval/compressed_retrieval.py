"""Does 4x KV compression cost RETRIEVAL, or only perplexity?

Every Stage D number in this project is perplexity. That is a real gap, because
this model has repeatedly shown the two moving independently:

  * the shipped RoPE ladder had the WORST retr@16384 of all 13 init-ablation
    arms (11.986) while looking unremarkable on perplexity;
  * the depth x length grid showed NoPE at parity-or-better at the LONGEST
    lookback and collapsing only at the ~13k gap.

So "4x KV costs ~4.8% perplexity" does not license any claim about needle
retrieval, and a latent that drops 75% of the KV rank is exactly the kind of
change that could cost retrieval while barely touching perplexity.

Compares the compressed arm against its MATCHED uncompressed control -- the same
data, schedule, learning rate and trainable surface -- so the difference is the
compression and nothing else. Both arms are loaded through the identical module
construction sequence used at training time; any deviation would silently change
which tensors the checkpoint binds to.
"""
import sys, json, argparse, torch
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora
from mercurius.surgery.transmla import convert_to_mla
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.paths import CACHE_DIR, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
LORA_RULES = [
    ("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
    ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),
    ("linear_attn.out_proj", 16), ("linear_attn.in_proj_qkv", 16),
    ("mlp.gate_proj", 16), ("mlp.up_proj", 16), ("mlp.down_proj", 16),
    ("lm_head", 0), ("embed_tokens", 0),
]
# gap, not total length: retrieval() builds [lead][needle][gap][needle] and
# materializes full-vocab logits, so 16384 (~7.7 GiB) is the safe ceiling.
GAPS = [256, 1024, 4096, 16384]


def build(adapters, mla):
    """Mirror train_recovery.py's construction order exactly.

    There, inject_lora runs once inside the --init-adapters block, then
    convert_to_mla, then inject_lora again (id-deduped). The MLA latents are
    nn.Linear under self_attn.k_proj.latent, so the second injection wraps them
    too -- reproducing that ordering is what makes the checkpoint keys bind.
    """
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    for l in get_trunk(m).layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")            # NoPE
    inject_lora(m, LORA_RULES, verbose=False)
    if mla:
        covs = {int(k): v.cuda().float() for k, v in
                torch.load(str(CACHE_DIR / 'kv_covs.pt'),
                           map_location="cpu").items()}
        convert_to_mla(m, d_c=256, covs=covs, verbose=False)
    inject_lora(m, LORA_RULES, verbose=False)

    sd = torch.load(adapters, map_location="cpu")
    missing, unexpected = m.load_state_dict(
        {k: v.cuda() for k, v in sd.items()}, strict=False)
    real = [k for k in unexpected if "lora" in k or "latent" in k
            or "A_log" in k or "dt_bias" in k or "in_proj" in k]
    print(f"    loaded {len(sd)} tensors | unexpected {len(unexpected)} "
          f"(load-bearing: {len(real)})", flush=True)
    if real:
        raise RuntimeError(f"checkpoint keys with no home: {real[:4]} -- the "
                           f"build order does not match training")
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="step300")
    ap.add_argument("--out", default="logs/compressed_retrieval.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    arms = [("4x KV (compressed)",
             f"ckpt/adapters-dense5m-mla-{a.step}.pt", True),
            ("uncompressed control",
             f"ckpt/adapters-dense5m-ctrl-{a.step}.pt", False)]

    rows = {}
    for name, adp, mla in arms:
        print(f"\n=== {name} ===", flush=True)
        m = build(adp, mla)
        r = {"ppl2048": perplexity(m, ids, 2048)}
        torch.cuda.empty_cache()
        r["ppl8192"] = perplexity(m, ids, 8192)
        torch.cuda.empty_cache()
        for gp in GAPS:
            try:
                _, _, r[f"gap{gp}"] = retrieval(m, ids, needle, gp)
            except RuntimeError as e:
                r[f"gap{gp}"] = float("nan")
                print(f"    gap {gp} failed: {str(e)[:60]}", flush=True)
            torch.cuda.empty_cache()
        rows[name] = r
        print(f"    ppl@2048 {r['ppl2048']:.3f}  ppl@8192 {r['ppl8192']:.3f}  "
              + "  ".join(f"gap{gp} {r[f'gap{gp}']:.3f}" for gp in GAPS), flush=True)
        del m
        torch.cuda.empty_cache()

    json.dump(rows, open(a.out, "w"), indent=2)
    c, u = rows["4x KV (compressed)"], rows["uncompressed control"]
    print("\n=== cost of 4x KV: perplexity vs retrieval ===")
    print(f"  {'metric':<14}{'compressed':>12}{'control':>11}{'cost':>11}")
    for k, lo_is_good in (("ppl2048", True), ("ppl8192", True)):
        print(f"  {k:<14}{c[k]:>12.3f}{u[k]:>11.3f}{100*(c[k]-u[k])/u[k]:>10.2f}%")
    for gp in GAPS:
        k = f"gap{gp}"
        if c[k] == c[k] and u[k] == u[k]:
            # retrieval gain: HIGHER is better, so cost is the drop
            print(f"  retr gap{gp:<6}{c[k]:>12.3f}{u[k]:>11.3f}"
                  f"{100*(u[k]-c[k])/abs(u[k]):>10.2f}%")
    print("\n  perplexity cost is ~4.8%. If the retrieval cost is much larger,")
    print("  the perplexity number has been understating what compression costs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
