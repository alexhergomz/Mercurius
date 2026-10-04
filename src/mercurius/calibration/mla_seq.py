"""SEQUENTIAL MLA calibration on the TRAINING MIX, on the fully built student.

WHY (user, 2026-09-29, for the long run): every calibration so far had one of two
defects. The stored CARE covariances (kv_covs_4b_mix.pt) came from data/calib_mix.txt
-- a 1.3 MB file concatenated by source, NOT the training mix, with no builder script
in the repo -- through care.build(), which installs NoPE and lacks the student's GDN-2 /
ScaleNorm state. The rank allocation inherited both. The step-0 recalibration measured
what the model half costs (-0.042 nats at 87.5%); the data half was never fixed.

WHAT THIS DOES
  1. windows: N random windows drawn from the ACTUAL TRAINING SOURCES in their training
     proportions -- an agent episode with probability episode_frac (rendered through
     the chat template exactly as training sees it), otherwise a window from the text
     corpus (document-aware when spans are known). Seeded, so it is reproducible.
  2. pass 0: per-layer input covariances on the UNCOMPRESSED student -> whitened spectra
     -> CARE water-filling of the total budget (allocate_ranks): rank goes where the
     whitened spectrum carries the most activation error.
  3. SEQUENTIAL conversion: layers in depth order; before converting layer l, its input
     covariance is RE-COLLECTED from the model with every earlier attention layer
     already compressed, so each latent is fitted to the inputs it will actually see
     (the SVD-LLM / GPTQ-style sequential update). Layer order matters only through
     those inputs: the factorisation of a layer depends on its covariance and W alone.
  4. outputs: the sequential covariances and the allocation, written as an ordinary
     covs file and an ungrouped groups JSON, so every harness (build(), ruler, longppl,
     bench_full) REPLAYS the calibration exactly through --covs / --mla-groups.
"""
import hashlib
import json
import math

import torch


def sample_calib_windows(train_ids, doc_spans, ep_ds, episode_frac, n, seq, seed=777,
                         min_ep_len=512, math_ds=None, math_frac=0.0):
    """n token windows (LongTensor, len <= seq) in the training mix's proportions --
    the same one-draw rule as the training loop: [0, ep) episode, [ep, ep+math) math,
    else a corpus window."""
    g = torch.Generator().manual_seed(seed)
    spans = [(s, e) for s, e in (doc_spans or []) if e - s >= seq] or None
    out, n_ep = [], 0
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        u = float(torch.rand(1, generator=g))
        src = (ep_ds if (ep_ds is not None and u < episode_frac) else
               math_ds if (math_ds is not None and episode_frac <= u < episode_frac + math_frac)
               else None)
        if src is not None:
            x = src.sample(g, seq, min_len=min_ep_len)
            if x is None:
                continue
            out.append(torch.as_tensor(x, dtype=torch.long)[:seq]); n_ep += 1
            continue
        if spans:
            s, e = spans[int(torch.randint(len(spans), (1,), generator=g))]
            o = int(torch.randint(s, e - seq + 1, (1,), generator=g))
        else:
            o = int(torch.randint(0, len(train_ids) - seq - 1, (1,), generator=g))
        out.append(train_ids[o:o + seq].clone())
    return out, n_ep


@torch.no_grad()
def collect_covs(model, windows, layers):
    """Input covariance of k_proj at each trunk layer in `layers` (fp64 accumulate)."""
    from mercurius.surgery.norm_fusion import get_trunk
    trunk = get_trunk(model)
    acc = {i: None for i in layers}
    cnt = {i: 0 for i in layers}

    def mk(i):
        def h(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            acc[i] = x.t() @ x if acc[i] is None else acc[i] + x.t() @ x
            cnt[i] += x.shape[0]
        return h
    hs = [trunk.layers[i].self_attn.k_proj.register_forward_hook(mk(i)) for i in layers]
    was = model.training
    model.eval()
    try:
        for w in windows:
            model(input_ids=w.reshape(1, -1).cuda(), logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
        model.train(was)
    return {i: (acc[i] / max(cnt[i], 1)).float() for i in layers}, cnt


def calibrate_mla_sequential(model, windows, budget, min_r=64, verbose=True, n_rope=0):
    """Water-filled allocation from pass-0 covariances, then sequential conversion.
    n_rope > 0: the ABSORBABLE variant (surgery/mla_rope.py, #68) -- same CARE whitening,
    water-filling and sequential re-collection, on the decoupled key space.
    Returns (sequential covariances {layer: cov}, allocation {layer: rank}, info)."""
    from mercurius.surgery.norm_fusion import get_trunk
    from mercurius.surgery.transmla import (allocate_ranks, convert_to_mla,
                                            whitened_spectra)
    if n_rope:
        from mercurius.surgery.mla_rope import decoupled_spectra, convert_to_mla_decoupled
        spectra = lambda m, c: decoupled_spectra(m, c, n_rope)
        convert = lambda m, alloc, covs, only: convert_to_mla_decoupled(
            m, alloc, covs, n_rope=n_rope, only=only, verbose=False)
    else:
        spectra = whitened_spectra
        convert = lambda m, alloc, covs, only: convert_to_mla(
            m, alloc=alloc, covs=covs, only=only, verbose=False)
    trunk = get_trunk(model)
    attn = [i for i, l in enumerate(trunk.layers) if hasattr(l, "self_attn")]
    covs0, cnt = collect_covs(model, windows, attn)
    covs0 = {i: c.cuda() for i, c in covs0.items()}
    alloc = allocate_ranks(spectra(model, covs0), int(budget), min_r=min_r)
    if verbose:
        print(f"  MLA sequential calibration: {len(windows)} windows, "
              f"{cnt[attn[0]]:,} vectors per layer; water-filled budget {int(budget)}: "
              f"{dict(sorted(alloc.items()))}", flush=True)
    seq_covs, info = {}, []
    for j, i in enumerate(attn):
        c = covs0[i] if j == 0 else collect_covs(model, windows, [i])[0][i].cuda()
        seq_covs[i] = c
        info += convert(model, alloc, {i: c}, {i})
        if verbose:
            _, r, fr, e = info[-1]
            drift = float((c - covs0[i]).norm() / covs0[i].norm())
            print(f"    layer {i:>2}: rank {r:>4} / {fr}  energy kept {e:.4f}  "
                  f"input-cov drift from uncompressed {100*drift:.2f}%", flush=True)
    return seq_covs, alloc, info


def save_calibration(prefix, seq_covs, alloc, model, meta):
    """covs file + ungrouped groups JSON (every KV head in one group per layer)."""
    cfg = getattr(model.config, "text_config", model.config)
    heads = list(range(int(cfg.num_key_value_heads)))
    torch.save({int(k): v.cpu() for k, v in seq_covs.items()}, prefix + "_covs.pt")
    json.dump({"groups": {str(k): [[heads, int(v)]] for k, v in sorted(alloc.items())},
               "note": "sequential CARE calibration on the training mix "
                       "(src/mercurius/calibration/mla_seq.py)",
               "meta": meta},
              open(prefix + "_groups.json", "w"), indent=1)
    return prefix + "_covs.pt", prefix + "_groups.json"


def calib_key(meta):
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:10]
