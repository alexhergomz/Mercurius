"""A tiny random Qwen3.5-architecture model (hybrid: Gated DeltaNet + gated attention, partial RoPE) with the real
Qwen3.5 tokenizer, so every pipeline stage runs on CPU in seconds. Set MERCURIUS_TEST_TOKENIZER to a directory
holding a Qwen3.5 tokenizer (default: the hub's Qwen/Qwen3.5-0.8B, downloaded once)."""
import os
import random
import sys

# CPU tests: hide fla so transformers' GatedDeltaNet and arch/gdn2 take their PyTorch paths (fla's Triton kernels
# need a GPU). Set MERCURIUS_TEST_FLA=1 to test with the kernels on a GPU machine.
if os.environ.get("MERCURIUS_TEST_FLA") != "1":
    for _m in ("fla", "fla.ops", "fla.ops.gdn2", "fla.modules", "fla.modules.convolution", "causal_conv1d"):
        sys.modules[_m] = None

import pytest
import torch

TOK = os.environ.get("MERCURIUS_TEST_TOKENIZER", "Qwen/Qwen3.5-0.8B")


def tiny_config(**over):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    kw = dict(vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
              linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
              layer_types=["linear_attention", "linear_attention", "full_attention", "full_attention"],
              rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                               "mrope_section": [1, 1, 2], "mrope_interleaved": True},
              tie_word_embeddings=True, max_position_embeddings=4096, rms_norm_eps=1e-6)
    kw.update(over)
    return Qwen3_5TextConfig(**kw)


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(tiny_config(), dtype=torch.float32)
    with torch.no_grad():                      # non-trivial norms / decay so the exactness checks mean something
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or n.endswith("A_log") or n.endswith("dt_bias"):
                p.add_(0.3 * torch.randn_like(p))
    m.save_pretrained(d)
    AutoTokenizer.from_pretrained(TOK).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    rng = random.Random(0)
    words = "the model reads long documents and keeps every fact it needs in a small state while it writes".split()
    docs = [" ".join(rng.choice(words) for _ in range(rng.randint(300, 900))) for _ in range(12)]
    (d / "train.txt").write_text("\n\n".join(docs))
    (d / "eval.txt").write_text(" ".join(rng.choice(words) for _ in range(1500)))
    return str(d / "train.txt"), str(d / "eval.txt")


def smoke_config(tiny_dir, corpus, out, **over):
    from mercurius.config import load_config
    o = [f"student.model={tiny_dir}", "student.dtype=float32", f"run.out_dir={out}", "run.device=cpu",
         f"data.text={corpus[0]}", f"data.eval_text={corpus[1]}", "data.seq=96", "data.tokenized_cache=false",
         "convert.mla.budget=64", "convert.mla.rank_multiple=32", "convert.mla.min_rank=32",
         "convert.mla.calib_windows=4", "convert.mla.calib_seq=64", "convert.verify_tokens=48",
         "train.steps=4", "train.eval_every=2", "train.eval_lengths=[64]", "train.log_every=1", "train.save_every=2",
         "train.adapters.rank=16", "train.chunk=32", "train.grad_checkpoint=false", "train.amp=none",
         "train.guards.min_avail_gb=0", "train.guards.gpu_temp_pause=1000",
         "qat.start_step=3", "qat.kv.head_dim=32"]
    o += [f"{k}={v}" for k, v in over.items()]
    cfg = load_config(None, o)
    cfg.teachers = [dict(kind="hf", model="student", weight=1.0, load_in="float32", url=None, gguf=None, head=None,
                         max_len=None)]
    from mercurius.config import _wrap
    cfg.teachers = _wrap(cfg.teachers)
    return cfg
