"""End to end on a tiny random model (CPU): exact stages, MLA exactness at full rank, convert -> train (with QAT and
a self-distillation teacher) -> export -> reload, and resume."""
import os

import torch

from conftest import smoke_config, tiny_config


def _logits(m, ids):
    with torch.no_grad():
        return m(input_ids=ids.view(1, -1)).logits[0].float()


def test_exact_stages(tiny_dir):
    from mercurius.arch.model import load_hf
    from mercurius.convert.lift import fold_norms, lift_gdn2, to_scalenorm
    from mercurius.arch.attention import PlainAttention
    from mercurius.arch.common import trunk
    m = load_hf(tiny_dir, "float32", "cpu")
    ids = torch.randint(0, 1000, (40,))
    ref = _logits(m, ids)
    fold_norms(m, log=lambda *_: None)
    to_scalenorm(m, log=lambda *_: None)
    assert torch.allclose(_logits(m, ids), ref, atol=1e-4, rtol=1e-4)
    for expand in (False, True):
        mm = load_hf(tiny_dir, "float32", "cpu")
        lift_gdn2(mm, expand_qk_heads=expand, log=lambda *_: None)
        for i, layer in enumerate(trunk(mm).layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn = PlainAttention.from_hf(layer.self_attn, i)
        assert torch.allclose(_logits(mm, ids), ref, atol=1e-4, rtol=1e-4), f"expand={expand}"


def test_mla_is_exact_at_full_rank_and_all_rope_components(tiny_dir):
    from mercurius.arch.model import load_hf
    from mercurius.arch.common import trunk, text_config
    from mercurius.convert.mla import convert_layer
    m = load_hf(tiny_dir, "float32", "cpu")
    ids = torch.randint(0, 1000, (40,))
    ref = _logits(m, ids)
    cfg = text_config(m)
    for i, layer in enumerate(trunk(m).layers):
        if hasattr(layer, "self_attn"):
            sa = layer.self_attn
            cov = torch.eye(cfg.hidden_size) + 0.1
            layer.self_attn, _ = convert_layer(sa, cov, 64, cfg.num_key_value_heads, cfg, i, per_head_q=True)
    assert torch.allclose(_logits(m, ids), ref, atol=1e-4, rtol=1e-4)


def test_convert_train_export_reload(tiny_dir, corpus, tmp_path):
    from mercurius.convert.pipeline import convert
    from mercurius.train.trainer import Trainer
    from mercurius.export import export, build_trained
    from mercurius.arch.model import load_model
    cfg = smoke_config(tiny_dir, corpus, str(tmp_path / "run"))
    convert(cfg, log=print)
    conv = load_model(os.path.join(cfg.run.out_dir, "converted"), "float32", "cpu")
    spec = conv.config.mercurius
    assert spec["kv_values_per_token"] > 0
    ranks = [s["self_attn"]["rank"] for s in spec["layers"].values() if "self_attn" in s]
    assert ranks and all(r % 32 == 0 for r in ranks)
    tr = Trainer(cfg, log=print)
    assert tr.run() == 0
    assert tr.qat_on
    out = export(cfg, log=print)
    exp = load_model(out, "float32", "cpu")
    assert exp.config.mercurius["quant"], "export wrote no quantized tensors"
    # the exported weights are the QAT forward's weights: same logits with the KV quantizer off on both sides
    trained = build_trained(cfg, log=lambda *_: None)
    from mercurius.quant.qat import install_qat
    install_qat(trained, cfg.qat, log=lambda *_: None)
    for mod in trained.modules():
        if hasattr(mod, "kv_quant"):
            mod.kv_quant = None
    ids = torch.randint(0, 1000, (48,))
    a, b = _logits(trained, ids), _logits(exp, ids)
    assert torch.allclose(a, b, atol=2e-3, rtol=2e-3), (a - b).abs().max()


def test_resume_continues(tiny_dir, corpus, tmp_path):
    from mercurius.convert.pipeline import convert
    from mercurius.train.trainer import Trainer
    out = str(tmp_path / "run2")
    cfg = smoke_config(tiny_dir, corpus, out, **{"qat.enabled": "false"})
    convert(cfg, log=lambda *_: None)
    t1 = Trainer(cfg, log=lambda *_: None)
    t1.end = 2                                               # stop early, as an interruption would
    t1.run()
    t2 = Trainer(cfg, log=lambda *_: None)
    assert t2.step == 2
    assert t2.run() == 0 and t2.step == 4
