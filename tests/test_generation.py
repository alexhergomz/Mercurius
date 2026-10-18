"""Incremental decoding through the transformers cache matches the full forward (converted model: GDN-2 recurrent +
conv state, MLA keys), on CPU (reference kernels) or GPU (fla kernels, MERCURIUS_TEST_FLA=1)."""
import os

import torch

from conftest import smoke_config


def test_cached_decode_matches_full_forward(tiny_dir, corpus, tmp_path):
    from mercurius.convert.pipeline import convert
    from mercurius.arch.model import load_model
    dev = "cuda" if os.environ.get("MERCURIUS_TEST_FLA") == "1" and torch.cuda.is_available() else "cpu"
    cfg = smoke_config(tiny_dir, corpus, str(tmp_path / "gen"), **{"run.device": dev})
    convert(cfg, log=lambda *_: None)
    m = load_model(os.path.join(cfg.run.out_dir, "converted"), "float32", dev)
    ids = torch.randint(0, 1000, (1, 30), device=dev)
    with torch.no_grad():
        full = m(input_ids=ids).logits[0].float()
        out = m(input_ids=ids[:, :20], use_cache=True)                 # prefill 20
        pkv = out.past_key_values
        steps = [out.logits[0, -1].float()]
        for t in range(20, 29):                                        # then one token at a time
            o = m(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            steps.append(o.logits[0, -1].float())
        chunk = m(input_ids=ids[:, :10], use_cache=True)               # chunked prefill: 10 + 20
        c2 = m(input_ids=ids[:, 10:30], past_key_values=chunk.past_key_values, use_cache=True)
    inc = torch.stack(steps)
    tol = 2e-3 if dev == "cuda" else 1e-4
    assert torch.allclose(inc, full[19:29], atol=tol, rtol=tol), (inc - full[19:29]).abs().max()
    assert torch.allclose(c2.logits[0].float(), full[10:30], atol=tol, rtol=tol)
