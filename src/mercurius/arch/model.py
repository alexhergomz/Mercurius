"""Loading and saving models: Hugging Face originals, converted checkpoints and exports.

A converted / exported checkpoint is a directory with
    config.json        the text model's transformers config + a "mercurius" section describing every replaced module
    model.safetensors  the weights (bf16, or packed QAT formats + "quant" metadata for exports)
    tokenizer files
and loads with `load_model(path)`: the transformers skeleton is built from the config, the converted modules are
swapped in from their specs, and the weights are loaded. No surgery is replayed, no covariance or rotation file is
needed, and nothing depends on the working directory (docs/porting_findings.md #9).
"""
from __future__ import annotations

import json
import os

import torch

from .attention import MLAAttention, PlainAttention
from .common import ScaleNorm, text_config, trunk
from .gdn2 import GatedDeltaNet2

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve(path_or_id, revision=None):
    if os.path.isdir(path_or_id):
        return path_or_id
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_id, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"])


def load_hf(path_or_id, dtype="bfloat16", device="cuda", revision=None):
    """The text model of a Hugging Face checkpoint (a multimodal checkpoint's vision tower is not loaded)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    path = resolve(path_or_id, revision)
    cfg = text_config(AutoConfig.from_pretrained(path))
    dt = DTYPES.get(dtype, dtype)
    m = AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=dt, device_map=device)
    return m.eval()


def load_tokenizer(path_or_id, revision=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(resolve(path_or_id, revision))


# ---------------------------------------------------------------------------------------------- spec
def model_spec(model):
    """Describe every module the conversion replaced, keyed by layer index."""
    layers = {}
    for i, layer in enumerate(trunk(model).layers):
        s = {}
        la = getattr(layer, "linear_attn", None)
        if isinstance(la, GatedDeltaNet2):
            s["linear_attn"] = {"kind": "gdn2", **la.spec()}
        sa = getattr(layer, "self_attn", None)
        if isinstance(sa, (MLAAttention, PlainAttention)):
            s["self_attn"] = sa.spec()
        sn = [n for n in ("input_layernorm", "post_attention_layernorm") if isinstance(getattr(layer, n), ScaleNorm)]
        if sn:
            s["scalenorm"] = sn
        if s:
            layers[str(i)] = s
    kv = sum(getattr(l.self_attn, "cache_values_per_token", lambda: 0)() for l in trunk(model).layers
             if hasattr(l, "self_attn"))
    return {"version": 2, "layers": layers, "kv_values_per_token": kv}


def apply_spec(model, spec):
    """Swap in the converted modules described by `spec` (weights are loaded afterwards)."""
    tr = trunk(model)
    H = text_config(model).hidden_size
    ref = next(model.parameters())
    for key, s in spec["layers"].items():
        i = int(key)
        layer = tr.layers[i]
        if "linear_attn" in s:
            layer.linear_attn = GatedDeltaNet2.from_spec(H, s["linear_attn"], i).to(ref.device, ref.dtype)
            layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
            layer.linear_attn.dt_bias.data = layer.linear_attn.dt_bias.data.float()
        if "self_attn" in s:
            cls = MLAAttention if s["self_attn"]["kind"] == "mla" else PlainAttention
            layer.self_attn = cls.from_spec(H, s["self_attn"], i).to(ref.device, ref.dtype)
            if cls is MLAAttention and layer.self_attn.q_map is not None:
                layer.self_attn.q_map.data = layer.self_attn.q_map.data.float()
        for n in s.get("scalenorm", []):
            old = getattr(layer, n)
            setattr(layer, n, ScaleNorm(getattr(old, "eps", getattr(old, "variance_epsilon", 1e-6)), 1.0, ref.device))
    return model


# ---------------------------------------------------------------------------------------------- save / load
def save_model(model, path, tokenizer=None, extra=None, state_dict=None):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    cfg = text_config(model).to_dict()
    cfg["mercurius"] = {**model_spec(model), **(extra or {})}
    cfg["architectures"] = [type(model).__name__]
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    sd = state_dict if state_dict is not None else model.state_dict()
    tied = getattr(text_config(model), "tie_word_embeddings", False)
    out, seen = {}, {}
    for k, v in sd.items():
        if tied and k.endswith("lm_head.weight"):
            continue
        if v.data_ptr() in seen and v.numel():                    # shared storage (e.g. the VeRA bank): once
            continue
        seen[v.data_ptr()] = k
        out[k] = v.detach().contiguous().cpu()
    save_file(out, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    return path


def load_model(path, dtype="bfloat16", device="cuda", strict=True):
    """A converted or exported checkpoint (or a plain Hugging Face one) as a ready-to-run model."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg_path = os.path.join(path, "config.json")
    raw = json.load(open(cfg_path))
    spec = raw.get("mercurius")
    if spec is None:
        return load_hf(path, dtype, device)
    cfg = AutoConfig.from_pretrained(path)
    dt = DTYPES.get(dtype, dtype)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dt)
    apply_spec(model, spec)
    quant = spec.get("quant", {})
    own = model.state_dict()
    loaded = set()
    packed = {}
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            base, _, field = k.rpartition(".")
            if base in quant:                                    # packed export: weight_packed / weight_scale ...
                packed.setdefault(base, {})[field] = f.get_tensor(k)
                continue
            if k not in own:
                if strict:
                    raise KeyError(f"unexpected tensor {k}")
                continue
            with torch.no_grad():
                own[k].copy_(f.get_tensor(k))
            loaded.add(k)
    if packed:
        from mercurius.quant.weights import FORMATS
        for base, t in packed.items():
            fmt, shape = quant[base]["format"], quant[base]["shape"]
            w = FORMATS[fmt]["unpack"]({kk: v.to(device) for kk, v in t.items()}, shape)
            key = base + ".weight"
            with torch.no_grad():
                own[key].copy_(w.to(own[key].dtype))
            loaded.add(key)
    missing = [k for k in own if k not in loaded and not k.endswith("lm_head.weight")]
    if missing and strict:
        raise KeyError(f"missing tensors: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if getattr(text_config(model), "tie_word_embeddings", False):
        model.tie_weights()
    return model.eval()
