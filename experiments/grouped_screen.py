"""Screen MLA factorisations at a FIXED total cache, before any training.

Arms (all on stage A+B + NoPE + GDN-2, bf16, no adapters -- so only the
factorisation differs; findings 0.10: this ranks initial factorisations, it
does not predict trained accuracy):

  full        no MLA (the reference every arm loses against)
  uniform     one joint latent per layer, budget / n_layers each (what we train)
  spectral    joint per layer, ranks water-filled across layers (allocate_ranks)
  grp_spec    KV heads grouped retrieval / other per layer, one latent per
              group, ranks water-filled over groups by whitened spectra
  grp_retr    same groups, water-filling weighted by the group's retrieval score

Metrics: wikitext ppl@8192 and RULER NIAH on the tasks that separate arms
(multikey_2, multikey_3, multivalue), value-token NLL and EM.

    python experiments/grouped_screen.py --budget 4096 --threshold 0.5
"""
import argparse
import json

import torch
from transformers import AutoTokenizer

from mercurius import guard
from mercurius.eval import ruler_gen as R
from mercurius.eval.retrieval_heads import (allocate_group_ranks, group_by_retrieval,
                                            group_spectra)
from mercurius.eval.ruler import score_sample
from mercurius.eval.suite import ce_and_topk
from mercurius.models.gdn2 import convert_to_gdn2
from mercurius.models.kda import load_kda_model
from mercurius.paths import CACHE_DIR, LOGS_DIR, STAGE_AB, WIKITEXT
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.surgery.transmla import allocate_ranks, convert_to_mla, whitened_spectra

TASKS = ["niah_multikey_2", "niah_multikey_3", "niah_multivalue"]


def base():
    m = load_kda_model(str(STAGE_AB), dtype=torch.bfloat16)
    install_rope_dial(m, 0, "global")
    convert_to_gdn2(m, verbose=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--scores", default=str(LOGS_DIR / "retrieval_heads_4b.json"))
    ap.add_argument("--covs", default=str(CACHE_DIR / "kv_covs_4b.pt"))
    ap.add_argument("--out", default=str(LOGS_DIR / "grouped_screen.json"))
    a = ap.parse_args()
    guard.cap_cuda_memory(60)
    pacer = guard.ThermalPacer(84.0, 80.0, 90.0, 85.0)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    ids = tok(open(WIKITEXT).read(), return_tensors="pt").input_ids[0]
    data = {(t, n): R.generate(t, n, a.samples, tok) for t in TASKS for n in a.lengths}
    covs = {int(k): v.cuda().float() for k, v in torch.load(a.covs).items()}
    sc = json.load(open(a.scores))
    n_kv = sc["n_kv"]
    kv = {(int(k.split(".")[0]), int(k.split(".")[1])): v for k, v in sc["kv"].items()}

    # allocations, computed once on an unconverted model
    m = base()
    cfg = getattr(m.config, "text_config", m.config)
    layers = sorted(covs)
    uniform = {l: a.budget // len(layers) for l in layers}
    spectral = allocate_ranks(whitened_spectra(m, covs), a.budget)
    grouping = group_by_retrieval(kv, n_kv, a.threshold)
    gs = group_spectra(m, covs, grouping, cfg.head_dim)
    gw = {(l, gi): max(sum(kv[(l, h)] for h in heads) / len(heads), 0.05)
          for l, groups in grouping.items() for gi, heads in enumerate(groups)}
    r_spec = allocate_group_ranks(gs, a.budget)
    r_retr = allocate_group_ranks(gs, a.budget, weights=gw)
    mk = lambda r: {l: [(heads, r[(l, gi)]) for gi, heads in enumerate(g)]
                    for l, g in grouping.items()}
    del m; torch.cuda.empty_cache()
    arms = {"full": None, "uniform": ("alloc", uniform), "spectral": ("alloc", spectral),
            "grp_spec": ("groups", mk(r_spec)), "grp_retr": ("groups", mk(r_retr))}
    print(f"grouping (threshold {a.threshold}, retrieval group first): "
          f"{ {l: g for l, g in grouping.items()} }")
    for name in ("spectral", "grp_spec", "grp_retr"):
        spec = arms[name][1]
        print(f"  {name:<9} " + (str(spec) if name == "spectral" else
              str({l: [r for _, r in g] for l, g in spec.items()})), flush=True)

    rows = {}
    for name, spec in arms.items():
        m = base()
        if spec is not None:
            kind, val = spec
            if kind == "alloc":
                convert_to_mla(m, covs=covs, alloc=val, verbose=False)
            else:
                convert_to_mla(m, covs=covs, groups=val, verbose=False)
        pacer.attach(m)
        with torch.no_grad():
            ppl = ce_and_topk(m, ids, 8192)["ppl"]
            res = {"ppl8192": ppl}
            for (t, n), samples in data.items():
                nll, preds, refs = [], [], []
                for s in samples:
                    _, nv, pred = score_sample(m, tok, s, -1)
                    nll.append(nv); preds.append(pred); refs.append(s["outputs"])
                res[f"{t}@{n}"] = {"nll_v": sum(nll) / len(nll),
                                   "em": R.string_match_all(preds, refs)}
        rows[name] = res
        print(f"  {name:<9} ppl@8192 {ppl:7.3f}  " + "  ".join(
            f"{t[5:]}@{n//1024}k EM {res[f'{t}@{n}']['em']:5.1f} nll {res[f'{t}@{n}']['nll_v']:.3f}"
            for t in TASKS for n in a.lengths), flush=True)
        json.dump({"grouping": {str(l): g for l, g in grouping.items()},
                   "alloc": {k: (v[1] if v else None) for k, v in arms.items()
                             if k in ("uniform", "spectral")},
                   "rows": rows}, open(a.out, "w"), indent=1, default=str)
        pacer.detach(); del m; torch.cuda.empty_cache()
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
