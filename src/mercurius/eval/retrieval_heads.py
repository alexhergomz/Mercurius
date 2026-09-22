"""Score the attention heads by how much they actually retrieve, then spend the
MLA latent budget where retrieval happens.

Motivation. The six full-attention layers were all compressed to the same
d_c = 256. A uniform split assumes every layer contributes equally to recall, and
the retrieval-head literature says it does not: in most models a small subset of
heads does the copying and the rest attend locally (Wu et al., arXiv:2404.15574).
RazorAttention (2407.15891) and DuoAttention (2410.10819) keep those heads at
full precision and compress the others; Palu (2407.21118) reports that per-head
low-rank decomposition (M-LRD) degrades and that grouping heads (G-LRD) is the
usable form.

Granularity. This model has 8 query heads but only 2 KV heads per layer, and the
latent replaces the KV projections. Grouping inside a layer is therefore already
per-KV-head, which is the variant Palu found degrades. The axis with room left is
ACROSS layers: six layers share a total budget of 6 * 256 = 1536, and it is
currently split evenly.

Score. A head's retrieval score is the fraction of answer-token positions whose
argmax attention lands inside the needle span, teacher-forced (no generation, so
the measurement is deterministic). A layer's score is the max over its heads --
one strong retrieval head is enough to make a layer worth paying for.
"""
import re, torch


NEEDLE = "The special access code for {topic} is {code}."
QUESTION = ("\n\nQuestion: what is the special access code for {topic}?"
            "\nAnswer: The special access code for {topic} is")


def build_sample(tok, filler_text, topic, code, length, depth=0.5):
    """Filler with one needle at a fractional depth, then the question."""
    q = QUESTION.format(topic=topic)
    n = NEEDLE.format(topic=topic, code=code)
    q_ids = tok(q, add_special_tokens=False).input_ids
    n_ids = tok(n, add_special_tokens=False).input_ids
    room = length - len(q_ids) - len(n_ids)
    if room < 64:
        raise ValueError(f"length {length} too small for the needle plus question")
    f_ids = tok(filler_text, add_special_tokens=False).input_ids[:room]
    cut = int(len(f_ids) * depth)
    ids = f_ids[:cut] + n_ids + f_ids[cut:] + q_ids
    needle_lo = cut
    needle_hi = cut + len(n_ids)
    ans_ids = tok(f" {code}", add_special_tokens=False).input_ids
    return (torch.tensor(ids + ans_ids).unsqueeze(0),
            needle_lo, needle_hi, len(ids), len(ans_ids))


@torch.no_grad()
def attn_layer_indices(model):
    """True trunk indices of the full-attention layers, e.g. [3,7,11,15,19,23].

    `out.attentions` carries one entry per ATTENTION layer, so enumerating it
    gives 0..5, not the trunk index. Scores keyed 0..5 and spectra keyed by trunk
    index silently agree on exactly one key, 3: weighting the allocator with such
    a dict gave layer 3 its real score and every other layer the floor, a 20x
    advantage that looked like a finding (693 of 1536 ranks) and was a key bug.
    """
    from mercurius.surgery.norm_fusion import get_trunk
    return [i for i, l in enumerate(get_trunk(model).layers)
            if getattr(l, "self_attn", None) is not None]


def score_heads(model, tok, filler_text, lengths=(4096,), depths=(0.25, 0.5, 0.75),
                n_samples=2, seed=0, verbose=True):
    """Per-(layer, head) retrieval score. Eager attention, one sample at a time.

    Memory is the reason for the shape of this loop: output_attentions keeps an
    L x L map per head per attention layer, which at L = 4096 is 8 * 4096^2 * 4 B
    = 537 MB for ONE layer. Scores are reduced to scalars and the maps freed
    before the next sample, so peak stays at one sample's worth.
    """
    import random
    rng = random.Random(seed)
    attn_idx = attn_layer_indices(model)
    topics = ["Helsinki", "the north archive", "project Vesta", "the blue ledger",
              "Marseille", "the tin cabinet"]
    hits, tot = {}, {}
    for L in lengths:
        for depth in depths:
            for s in range(n_samples):
                topic = topics[rng.randrange(len(topics))]
                code = "".join(rng.choice("0123456789abcdef") for _ in range(6))
                x, lo, hi, n_ctx, n_ans = build_sample(
                    tok, filler_text, topic, code, L, depth)
                out = model(input_ids=x.to(model.device), output_attentions=True,
                            use_cache=False)
                slot = -1
                for att in out.attentions:
                    if att is None:
                        continue
                    slot += 1
                    li = attn_idx[slot]   # trunk index, matching whitened_spectra
                    # positions that predict the answer tokens
                    a = att[0, :, n_ctx - 1:n_ctx - 1 + n_ans, :]   # H x n_ans x L
                    arg = a.argmax(dim=-1)                          # H x n_ans
                    inside = ((arg >= lo) & (arg < hi)).float().sum(dim=-1)
                    for h in range(inside.numel()):
                        hits[(li, h)] = hits.get((li, h), 0.0) + float(inside[h])
                        tot[(li, h)] = tot.get((li, h), 0) + n_ans
                del out
                torch.cuda.empty_cache()
    scores = {k: hits[k] / tot[k] for k in hits}
    if verbose:
        by_layer = {}
        for (li, h), v in sorted(scores.items()):
            by_layer.setdefault(li, []).append(v)
        print("  retrieval score per attention layer, by TRUNK index (fraction "
              "of answer positions whose argmax attention is inside the needle)")
        for li in sorted(by_layer):
            hs = by_layer[li]
            print(f"    layer {li:>2}  max {max(hs):.3f}  mean {sum(hs)/len(hs):.3f}  "
                  f"heads " + " ".join(f"{v:.2f}" for v in hs), flush=True)
    return scores


def layer_scores(scores, agg="mean"):
    """Reduce per-head scores to one number per layer.

    `max` was the first choice, on the reasoning that a single strong retrieval
    head makes a layer worth paying for. Measured, it saturates and destroys the
    signal: layer maxima are 0.80, 1.00, 1.00, 1.00, 1.00, 0.927, a spread of
    1.25x, because almost every layer owns at least one head that tracks the
    needle perfectly. Weighting an allocator by that is nearly uniform weighting,
    and would have produced a null screen that looked like a real negative.

    The mean over heads spreads 3.62x (0.239 to 0.864) and says something
    different and more useful: what FRACTION of a layer's heads retrieve, which
    is what determines how much of its KV subspace carries retrieval information
    and therefore how much rank it needs.
    """
    by = {}
    for (li, h), v in scores.items():
        by.setdefault(li, []).append(v)
    if agg == "max":
        return {l: max(v) for l, v in by.items()}
    return {l: sum(v) / len(v) for l, v in by.items()}


def allocate_ranks_retrieval(spectra, lscores, total_budget, min_r=64, floor=0.05):
    """Water-filling as in allocate_ranks, but the marginal gain of a rank is
    weighted by the layer's retrieval score.

    allocate_ranks minimises total truncation error, which treats all layers as
    equally worth reconstructing. If retrieval is concentrated, reconstruction
    error in a non-retrieval layer is cheaper than the same error in a retrieval
    layer, so the priority is score * next-singular-value-squared. `floor` keeps
    a zero-scoring layer from being starved below its share of the spectrum.
    """
    import heapq
    missing = [k for k in spectra if k not in lscores]
    if missing:
        raise KeyError(
            f"no retrieval score for layers {missing}; scores cover "
            f"{sorted(lscores)} but the spectra are keyed {sorted(spectra)}. "
            "Falling back to `floor` for a missing layer silently turns this "
            "into a near-uniform weighting with one accidental winner -- the "
            "bug that produced 693 of 1536 ranks for layer 3.")
    ranks = {k: min_r for k in spectra}
    used = sum(ranks.values())
    w = {k: max(float(lscores[k]), floor) for k in spectra}

    def prio(k, r):
        S2 = spectra[k].pow(2)
        if r >= S2.numel() - 1:
            return -1.0
        return w[k] * float(S2[r])

    heap = [(-prio(k, ranks[k]), k) for k in spectra]
    heapq.heapify(heap)
    while used < total_budget and heap:
        negp, k = heapq.heappop(heap)
        if -negp <= 0:
            continue
        ranks[k] += 1
        used += 1
        heapq.heappush(heap, (-prio(k, ranks[k]), k))
    return ranks


def group_by_retrieval(kv_scores, n_kv, threshold):
    """Per layer, split KV heads into [retrieval heads, the rest] by score.

    kv_scores: {(layer, kv_head): score}. A layer whose heads all fall on one
    side gets a single group. Returns {layer: [heads, ...]}, retrieval first.
    """
    out = {}
    for li in sorted({l for l, _ in kv_scores}):
        r = [g for g in range(n_kv) if kv_scores[(li, g)] >= threshold]
        o = [g for g in range(n_kv) if kv_scores[(li, g)] < threshold]
        out[li] = [x for x in (r, o) if x]
    return out


@torch.no_grad()
def group_spectra(model, covs, grouping, head_dim):
    """Whitened singular values of each group's own [K; V] rows -- what
    LatentKV._init_grouped truncates, so allocation and factorisation agree."""
    import math
    from mercurius.surgery.norm_fusion import get_trunk
    from mercurius.surgery.transmla import merged_weight, whiten_factor
    out = {}
    for li, groups in grouping.items():
        sa = get_trunk(model).layers[li].self_attn
        Wk, _ = merged_weight(sa.k_proj)
        Wv, _ = merged_weight(sa.v_proj)
        sk = Wk.norm() / math.sqrt(Wk.numel()); sv = Wv.norm() / math.sqrt(Wv.numel())
        gm = (sk * sv).sqrt()
        L = whiten_factor(covs[li].to(torch.float32).to(Wk.device))
        for gi, heads in enumerate(groups):
            rows = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim)
                              for h in heads]).to(Wk.device)
            W = torch.cat([Wk[rows] * (gm / sk), Wv[rows] * (gm / sv)], 0)
            out[(li, gi)] = torch.linalg.svdvals(W @ L)
    return out


def allocate_group_ranks(spectra, total_budget, weights=None, min_r=16):
    """Water-filling over (layer, group) units: each extra rank goes where
    weight * next-singular-value^2 is largest. weights=None is the pure
    spectral objective (minimum total whitened truncation error); retrieval
    weights spend more of the same budget on retrieval groups."""
    import heapq
    w = {k: (1.0 if weights is None else float(weights[k])) for k in spectra}
    ranks = {k: min(min_r, spectra[k].numel()) for k in spectra}
    used = sum(ranks.values())

    def prio(k):
        S2 = spectra[k].pow(2)
        return -1.0 if ranks[k] >= S2.numel() else w[k] * float(S2[ranks[k]])

    heap = [(-prio(k), k) for k in spectra]
    heapq.heapify(heap)
    while used < total_budget and heap:
        negp, k = heapq.heappop(heap)
        if -negp <= 0:
            continue
        ranks[k] += 1
        used += 1
        heapq.heappush(heap, (-prio(k), k))
    return ranks


if __name__ == "__main__":
    import argparse, sys, re as _re
    from transformers import AutoTokenizer
    from mercurius.models.kda import load_kda_model
    from mercurius.surgery.transmla import whitened_spectra, allocate_ranks
    from mercurius.recovery.train import CKPT
    from mercurius.paths import STAGE_AB, CACHE_DIR, LOGS_DIR, FINEWEB_LONG

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(STAGE_AB))
    ap.add_argument("--covs", default=str(CACHE_DIR / "kv_covs.pt"))
    ap.add_argument("--filler", default=str(FINEWEB_LONG))
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--budget", type=int, default=1536)
    ap.add_argument("--score-agg", choices=["mean", "max"], default="mean",
                    help="reduce per-head scores to a layer score. max saturates "
                         "(1.25x spread) and is kept only for comparison")
    ap.add_argument("--out", default=str(LOGS_DIR / "retrieval_heads.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT)
    raw = open(a.filler, encoding="utf-8", errors="replace").read(2_000_000)
    filler = _re.sub(r"\s+", " ", raw)

    # bf16 for the scoring pass: argmax of an attention row is what is measured,
    # and it is insensitive to that precision, while fp32 attention maps at
    # L = 4096 would be 1.07 GB per layer instead of 537 MB.
    model = load_kda_model(a.ckpt, dtype=torch.bfloat16)
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"
    model.eval()

    scores = score_heads(model, tok, filler, lengths=(a.length,),
                         n_samples=a.samples)
    ls = layer_scores(scores, agg=a.score_agg)
    print(f"  layer scores (max over heads): "
          f"{ {k: round(v, 3) for k, v in sorted(ls.items())} }", flush=True)

    covs = torch.load(a.covs, map_location="cpu")
    spectra = whitened_spectra(model, covs)
    uni = {k: a.budget // len(spectra) for k in spectra}
    spec = allocate_ranks(spectra, a.budget)
    retr = allocate_ranks_retrieval(spectra, ls, a.budget)
    print(f"  d_c at a fixed total of {a.budget}:")
    print(f"    uniform    {dict(sorted(uni.items()))}")
    print(f"    spectral   {dict(sorted(spec.items()))}")
    print(f"    retrieval  {dict(sorted(retr.items()))}")

    import json
    json.dump({"scores": {f"{l}.{h}": v for (l, h), v in scores.items()},
               "layer_scores": ls, "uniform": uni,
               "spectral": spec, "retrieval": retr},
              open(a.out, "w"), indent=1)
    print(f"  -> {a.out}")
