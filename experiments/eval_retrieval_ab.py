"""Retrieval A/B between two recovery checkpoints.

The training loop reports perplexity only. Retrieval is the metric the surgery
actually damages -- linear attention replacing softmax, a 4x compressed KV
cache, no positional encoding are all long-range changes -- and it is measured
nowhere in the loop.

ALWAYS include the ORIGINAL arm. The question this project answers is whether a
converted model matches the model it was carved out of, and only that comparison
addresses it. Differences between two converted checkpoints describe adapter
configuration; treating one of them as the reference turns an internal ablation
into an apparent shortfall. Measured on longdoc, the converted model is ahead of
the original on perplexity AND retrieval at a 4x smaller KV cache -- a fact that
was obscured for some time by comparing converted models against each other.

Secondary question, and the reason for the data arms: every run before
2026-09-13 trained on a corpus whose median document is 550 tokens, sampled as
one concatenated stream, so an 8192-token window held about 15 unrelated
documents and no dependency longer than roughly 2k.

Reports gain = NLL(first occurrence) - NLL(second occurrence) at a set of gaps.
Higher is better: it is how much cheaper the needle becomes once the model has
already seen it, which is exactly what retrieval buys.
"""
import argparse
import json
import sys

import torch
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.surgery.transmla import convert_to_mla
from mercurius.adapters.lora import inject_lora, freeze_base
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.recovery.train import CKPT, EVAL_DATA
from mercurius.paths import CACHE_DIR

GAPS = [1024, 4096, 16384]
LENGTHS = [2048, 8192]


def rules_from_checkpoint(sd):
    """Derive the LoRA rules from the checkpoint's own tensor shapes.

    The evaluator must NOT import LORA_RULES. That constant tracks whatever the
    current recipe uses, so raising the FFN rank to 32 immediately broke loading
    for every rank-16 checkpoint on disk -- strict=False forgives missing keys
    but not mismatched shapes, so it failed loudly rather than silently, which
    is the only reason this was cheap to find.

    A checkpoint records its own structure: each `<module>.lora_A` has shape
    (rank, in_features). Reading it back means old checkpoints stay loadable
    however the recipe moves.
    """
    rules = {}
    for k, v in sd.items():
        if not k.endswith(".lora_A"):
            continue
        path = k[: -len(".lora_A")]
        if path.endswith(".base"):          # inner adapter of a double wrap
            path = path[: -len(".base")]
        pat = ".".join(path.split(".")[-2:])
        rules[pat] = int(v.shape[0])
    return sorted(rules.items(), key=lambda kv: -len(kv[0]))


def build_original():
    """The unmodified teacher, no surgery and no adapters.

    This is the only comparison that settles whether the method works. Gaps
    between two converted models answer a question about adapter configuration,
    not about the result.
    """
    from transformers import AutoModelForCausalLM
    from mercurius.recovery.train import ORIG
    m = AutoModelForCausalLM.from_pretrained(ORIG, dtype=torch.bfloat16,
                                             device_map="cuda")
    return m.eval()


def build(adapters, dc, covs_path, double_adapter=False):
    """Reconstruct a trained model.

    double_adapter reproduces the pre-2026-09-13 injection, which wrapped every
    target twice. Checkpoints from that era carry both adapters per target, and
    a single injection leaves the inner ones unmatched -- strict=False drops
    them in silence and the model reads far worse than the run reported.
    """
    sd = torch.load(adapters, map_location="cpu")
    rules = rules_from_checkpoint(sd)
    by_r = {}
    for _, r in rules:
        by_r[r] = by_r.get(r, 0) + 1
    print(f"    ranks from checkpoint: "
          f"{', '.join(f'{c} pattern(s)@r{r}' for r, c in sorted(by_r.items()))}",
          flush=True)
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    trunk = m.model.language_model if hasattr(m.model, "language_model") else m.model
    for l in trunk.layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")
    inject_lora(m, rules, verbose=False)
    if double_adapter:
        inject_lora(m, rules, verbose=False)
    freeze_base(m)
    if dc:
        covs = {int(k): v.cuda().float()
                for k, v in torch.load(covs_path, map_location="cpu").items()}
        convert_to_mla(m, d_c=dc, covs=covs, verbose=False)
    missing = m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    unexpected = [k for k in sd if k not in dict(m.named_parameters())
                  and k not in dict(m.named_buffers())]
    if unexpected:
        print(f"    WARNING {len(unexpected)} checkpoint tensors had no home in "
              f"the model (e.g. {unexpected[0]}) -- the rebuild does not match "
              f"the run", flush=True)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="tag=path[:double] entries")
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--covs", default=str(CACHE_DIR / 'kv_covs.pt'))
    ap.add_argument("--out", default="logs/retrieval_ab.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(EVAL_DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    hdr = f"{'arm':<14}" + "".join(f"{'ppl@'+str(n):>10}" for n in LENGTHS) \
          + "".join(f"{'gain@'+str(k//1024)+'k':>11}" for k in GAPS)
    print(hdr); print("-" * len(hdr))
    rows = []
    for spec in a.arms:
        tag, rest = spec.split("=", 1)
        if rest == "ORIGINAL":
            m = build_original()
            row = {"arm": tag, "ppl": {}, "gain": {}}
            for n in LENGTHS:
                row["ppl"][n] = perplexity(m, ids, n); torch.cuda.empty_cache()
            for k in GAPS:
                row["gain"][k] = retrieval(m, ids, needle, k)[2]; torch.cuda.empty_cache()
            print(f"{tag:<14}" + "".join(f"{row['ppl'][n]:>10.3f}" for n in LENGTHS)
                  + "".join(f"{row['gain'][k]:>11.3f}" for k in GAPS), flush=True)
            rows.append(row); del m; torch.cuda.empty_cache()
            continue
        dbl = rest.endswith(":double")
        path = rest[:-7] if dbl else rest
        m = build(path, a.dc, a.covs, double_adapter=dbl)
        row = {"arm": tag, "ppl": {}, "gain": {}}
        for n in LENGTHS:
            row["ppl"][n] = perplexity(m, ids, n)
            torch.cuda.empty_cache()
        for k in GAPS:
            row["gain"][k] = retrieval(m, ids, needle, k)[2]
            torch.cuda.empty_cache()
        print(f"{tag:<14}" + "".join(f"{row['ppl'][n]:>10.3f}" for n in LENGTHS)
              + "".join(f"{row['gain'][k]:>11.3f}" for k in GAPS), flush=True)
        rows.append(row)
        del m
        torch.cuda.empty_cache()

    if len(rows) > 1:
        base = rows[0]
        print("\n  vs the first arm (gain: higher is better):")
        for r in rows[1:]:
            dp = "".join(f"{(r['ppl'][n]/base['ppl'][n]-1)*100:>+9.2f}%" for n in LENGTHS)
            dg = "".join(f"{r['gain'][k]-base['gain'][k]:>+10.3f}" for k in GAPS)
            print(f"  {r['arm']:<14}{dp}{dg}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
