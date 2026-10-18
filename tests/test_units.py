import math

import pytest
import torch

from mercurius.quant import weights as W
from mercurius.quant.kv import FoveaKV, TurboQuantKV, IntKV, nested_codebook, rht


def test_q4_0_matches_llama_rounding_and_roundtrips():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q, d = W.q4_0_codes(w)
    g = w.view(64, 8, 32)
    # llama.cpp quantize_row_q4_0_ref: d = max / -8 (max = signed absmax), q = min(15, (int8)(x / d + 8.5))
    mx = g.gather(-1, g.abs().argmax(-1, keepdim=True)).squeeze(-1)
    dd = (mx / -8).half().float()
    ref = torch.clamp(torch.trunc(g / dd.unsqueeze(-1) + 8.5), max=15)
    assert torch.equal(q.float(), ref)
    packed = W.q4_0_pack(w)
    assert torch.equal(W.q4_0_unpack(packed, w.shape), W.q4_0_fake(w).float())


@pytest.mark.parametrize("fmt", ["q4_0", "iq4_nl", "q8_0"])
def test_formats_pack_equals_fake_and_error_is_small(fmt):
    torch.manual_seed(1)
    w = torch.randn(96, 128) * 0.02
    f = W.fake(w, fmt)
    u = W.FORMATS[fmt]["unpack"](W.FORMATS[fmt]["pack"](w), w.shape)
    assert torch.equal(u, f.float())
    err = ((f - w).norm() / w.norm()).item()
    assert err < {"q4_0": 0.12, "iq4_nl": 0.11, "q8_0": 0.01}[fmt]


def test_iq4_nl_beats_q4_0_on_gaussian_weights():
    torch.manual_seed(2)
    w = torch.randn(256, 512)
    e = {f: ((W.fake(w, f) - w).norm() / w.norm()).item() for f in ("q4_0", "iq4_nl")}
    assert e["iq4_nl"] < e["q4_0"]


def test_ste_value_and_gradient():
    w = torch.randn(8, 64, requires_grad=True)
    y = W.ste(w, "q4_0")
    assert torch.equal(y.detach(), W.fake(w.detach(), "q4_0"))
    y.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_rht_is_orthonormal_and_codebook_is_nested():
    R = rht(128, 0)
    assert torch.allclose(R @ R.T, torch.eye(128), atol=1e-5)
    bnd, lut16, lut4, mf, mc = nested_codebook()
    assert len(lut16) == 16 and len(lut4) == 4 and mf < mc < 0.2


@pytest.mark.parametrize("bits,limit", [(4, 0.15), (2, 0.40)])
def test_foveakv_error_and_shapes(bits, limit):
    torch.manual_seed(3)
    x = torch.randn(2, 300, 256) + 0.5
    q = FoveaKV(bits=bits, head_dim=128)
    y = q.quantize(x)
    assert y.shape == x.shape
    assert ((y - x).norm() / x.norm()).item() < limit
    x.requires_grad_(True)
    q(x).sum().backward()                                   # straight-through
    assert torch.equal(x.grad, torch.ones_like(x))


def test_foveakv_window_keeps_tail_exact():
    x = torch.randn(1, 400, 128)
    y = FoveaKV(4, 128, window=128).quantize(x)
    keep = 400 - (400 - 128) // 128 * 128
    assert torch.equal(y[:, -keep:], x[:, -keep:]) and not torch.equal(y[:, :10], x[:, :10])


def test_other_kv_quantizers():
    x = torch.randn(1, 50, 96)
    for q in (TurboQuantKV(4), IntKV(4, 32)):
        assert ((q(x) - x).norm() / x.norm()).item() < 0.2


def test_water_filling_respects_multiples():
    from mercurius.convert.mla import allocate_ranks
    torch.manual_seed(0)
    spectra = {i: torch.sort(torch.rand(700) * (i + 1), descending=True).values for i in range(5)}
    r = allocate_ranks(spectra, 1280, step=128, min_rank=128)
    assert sum(r.values()) == 1280 and all(v % 128 == 0 for v in r.values())
    assert r[4] >= r[0]                                       # the larger spectrum gets more


def test_gdn2_reference_reduces_to_gated_delta_rule():
    """b = w = beta (tiled) must reproduce transformers' scalar gated delta rule."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    from mercurius.arch.gdn2 import gdn2_reference
    torch.manual_seed(4)
    B, T, H, Dk, Dv = 1, 7, 2, 8, 6
    q, k = torch.randn(B, T, H, Dk), torch.randn(B, T, H, Dk)
    v = torch.randn(B, T, H, Dv)
    g = -torch.rand(B, T, H)
    beta = torch.rand(B, T, H)
    ref, _ = torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                              use_qk_l2norm_in_kernel=True)
    out, _ = gdn2_reference(q, k, v, g.unsqueeze(-1).expand(B, T, H, Dk), beta.unsqueeze(-1).expand(B, T, H, Dk),
                            beta.unsqueeze(-1).expand(B, T, H, Dv))
    assert torch.allclose(out, ref.float(), atol=1e-5)


def test_config_rejects_typos_and_parses_overrides():
    from mercurius.config import load_config
    with pytest.raises(KeyError):
        load_config(None, ["student.model=x", "train.lrr=1"])
    cfg = load_config(None, ["student.model=x", "train.lr=1e-4", "data.length_mix={\"4096\": 0.5, \"8192\": 0.5}"])
    assert cfg.train.lr == 1e-4 and cfg.data.length_mix == {4096: 0.5, 8192: 0.5}
    assert cfg.train.taid.space == "prob"                    # reverse KL -> skew reverse (prob space)
