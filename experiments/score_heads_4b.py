"""Retrieval scores per query head AND per KV head, on the model as compressed.

Scored on stage A+B with the NoPE dial and the GDN-2 lift applied -- the exact
state the CARE covariances were collected on and the MLA factorization is
fitted to. Removing RoPE can change which heads retrieve, so scoring the
RoPE model would rank the wrong heads.

Per-KV-head scores are what grouping needs: the latent replaces the K/V
projections, so a group is a set of KV heads. Under GQA each KV head serves
n_heads / n_kv consecutive query heads (repeat_kv), and its score is the mean
over those query heads (findings 0.7: max saturates).

    python experiments/score_heads_4b.py --lengths 4096 8192 --samples 3
"""
import argparse
import json
import re

import torch
from transformers import AutoTokenizer

from mercurius import guard
from mercurius.eval.retrieval_heads import score_heads
from mercurius.models.gdn2 import convert_to_gdn2
from mercurius.models.kda import load_kda_model
from mercurius.paths import FINEWEB_LONG, LOGS_DIR, STAGE_AB
from mercurius.surgery.rope_dial import install_rope_dial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default=str(LOGS_DIR / "retrieval_heads_4b.json"))
    ap.add_argument("--filler", default=None, metavar="TXT",
                    help="haystack text. The default is FineWeb prose, which "
                         "scores which heads retrieve a needle FROM PROSE. Half "
                         "our training mix is code, and the heads that find a "
                         "symbol in a repository are not necessarily the ones "
                         "that find a fact in an article -- so the grouping and "
                         "rank allocation derived from prose scores carry the "
                         "same domain bias as prose-calibrated whitening.")
    a = ap.parse_args()
    guard.cap_cuda_memory(60)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    filler = re.sub(r"\s+", " ", open(a.filler or FINEWEB_LONG, encoding="utf-8",
                                      errors="replace").read(3_000_000))
    m = load_kda_model(str(STAGE_AB), dtype=torch.bfloat16)
    install_rope_dial(m, 0, "global")
    convert_to_gdn2(m, verbose=False)
    for mod in m.modules():
        if hasattr(mod, "config"):
            mod.config._attn_implementation = "eager"
    m.eval()
    cfg = getattr(m.config, "text_config", m.config)
    H, G = cfg.num_attention_heads, cfg.num_key_value_heads
    per = H // G

    with torch.no_grad():
        s = score_heads(m, tok, filler, lengths=tuple(a.lengths), n_samples=a.samples)
    kv = {}
    for (li, h), v in s.items():
        kv.setdefault((li, h // per), []).append(v)
    kv = {k: sum(v) / len(v) for k, v in kv.items()}
    print(f"\n  per-KV-head score (mean over its {per} query heads):")
    for li in sorted({k[0] for k in kv}):
        print(f"    layer {li:>2}  " + "  ".join(f"kv{g}={kv[(li, g)]:.3f}" for g in range(G)))
    json.dump({"q": {f"{l}.{h}": v for (l, h), v in s.items()},
               "kv": {f"{l}.{g}": v for (l, g), v in kv.items()},
               "n_heads": H, "n_kv": G, "lengths": a.lengths, "samples": a.samples},
              open(a.out, "w"), indent=1)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
