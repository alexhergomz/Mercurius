"""Run configuration: one YAML file describes a whole conversion (student, teachers, surgery, data, training,
QAT, export). Every key has a default here; a config file only states what differs. Unknown keys are an error
(a typo must not silently fall back to a default), and any value can be overridden on the command line:

    mercurius run configs/qwen3.5-4b.yaml train.lr=1e-4 teachers.0.url=http://host:8077
"""
from __future__ import annotations

import copy
import json
import os

DEFAULTS = {
    "run": {
        "name": "mercurius",
        "out_dir": "runs/{name}",          # every artefact of the run lives here
        "seed": 0,
        "device": "cuda",
    },
    # The model being converted. Any Hugging Face causal LM of the Qwen3 / Qwen3.5 family (hybrid Gated DeltaNet +
    # gated attention, or plain attention); a local directory or a hub id.
    "student": {
        "model": None,
        "dtype": "bfloat16",
        "revision": None,
    },
    # One or more teachers. kind:
    #   hf      a Hugging Face model loaded in this process (frozen). model: hub id / path; "student" = the student's
    #           ORIGINAL weights (self-distillation; loaded once, before surgery).
    #   server  a llama.cpp server returning final hidden states (--embeddings --pooling none --embd-normalize -1);
    #           its output head is read from `gguf` (or `head`, a .npy / .pt file).
    # weight: relative weight. combine: "loss" (weighted sum of per-teacher divergences) or "mixture" (one target,
    # the weighted mixture of the teachers' distributions). All teachers must share the student's vocabulary.
    "teachers": [],
    "teacher_defaults": {
        "kind": "hf",
        "model": None,
        "weight": 1.0,
        "load_in": "bfloat16",            # hf: bfloat16 | float16 | nf4 (bitsandbytes, for large teachers)
        "url": None,
        "gguf": None,
        "head": None,
        "max_len": None,                  # server: refuse windows longer than its -ub
    },
    "teacher_combine": "loss",
    "convert": {
        "fold_norms": True,               # RMSNorm gains folded into the consuming projections (exact)
        "scalenorm": True,                # folded norms -> one learned scalar each (exact at init)
        "gdn2": {
            "enabled": True,              # Gated DeltaNet -> GDN-2 (channel-wise decay / erase / write), exact lift
            "expand_qk_heads": True,      # give q/k one head per value head (exact): kernels need it anyway
            "seed_alpha": None,           # retarget the median decay (None = keep the trained decay)
        },
        "mla": {
            "enabled": True,              # absorbable MLA on the full-attention layers
            "budget": 4096,               # total latent width summed over the attention layers
            "rank_multiple": 128,         # every latent width a multiple of this (tensor cores, block formats)
            "min_rank": 128,
            "n_rope": 1,                  # RoRoPE components kept as the exact decoupled RoPE key (x rotary dims)
            "k_rms": "exact",             # "exact" (frozen original k_proj) or an int rank (low-rank RMS estimate)
            "calib_windows": 256,
            "calib_seq": 1024,
            "calib_seed": 777,
        },
        "per_head_q": True,               # per-head query maps after q_norm, kept in the RoPE commutant
        "verify_tokens": 256,             # exactness checks after each exact stage (0 = skip)
    },
    "data": {
        "text": None,                     # documents separated by blank lines (long documents preferred)
        "episodes": None,                 # {"path": jsonl, "frac": f}: chat episodes, assistant tokens supervised
        "math": None,                     # {"path": jsonl, "frac": f}
        "eval_text": None,                # held-out text for perplexity
        "seq": 8192,
        "length_mix": None,               # {4096: 0.32, 8192: 0.63, 32768: 0.05}; corpus windows only
        "length_mix_start": 0,
        "doc_aware": True,                # windows drawn inside one document
        "tokenized_cache": True,          # cache tokenized corpora under out_dir/cache
    },
    "train": {
        "steps": 6000,
        "lr": 3.0e-5,                     # dense trainables (latents, norms, decay)
        "adapter_lr": 1.0e-2,             # VeRA vectors
        "qmap_lr": 1.0e-3,                # per-head query maps
        "pct_start": 0.05,
        "optimizer": "nadamw",            # nadamw | adamw
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        # Mixed precision: "bf16" / "fp16" run the student forward and the loss under torch.autocast (fp16 adds a
        # gradient scaler); "none" computes in the weights' dtype. Trainables always keep fp32 masters; with
        # student.dtype float32 and amp bf16 the frozen weights are fp32 too (more memory, no bf16 storage error).
        "amp": "bf16",
        "adapters": {
            "kind": "vera",
            "rank": 1024,
            "d_init": 0.1,
            "targets": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                        "linear_attn.in_proj_a", "linear_attn.in_proj_be", "linear_attn.in_proj_bw",
                        "self_attn.q_proj", "self_attn.o_proj",
                        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        },
        "train_norms": True,
        "train_decay": True,              # A_log / dt_bias of the GDN-2 layers
        "divergence": "reverse",          # forward | reverse | js | jeffreys
        "reverse_weight": 1.0,            # jeffreys only
        "ce_beta": 1.0,                   # excess cross-entropy over the teacher on the true token
        "taid": {"enabled": True, "space": "auto", "start": 0.4, "end": 1.0, "alpha": 5.0e-4, "beta": 0.99},
        "chunk": 512,                     # positions per vocabulary chunk
        "grad_checkpoint": True,
        "ckpt_above": 0,                  # checkpoint only windows longer than this
        "anneal": None,                   # {"from": F, "to": N}: end at N with a cosine anneal from F
        "eval_every": 500,
        "eval_lengths": [2048, 8192],
        "log_every": 25,
        "save_every": 25,                 # resume state
        "keep_step_ckpts": True,
        "guards": {"mem_cap_gb": None, "min_avail_gb": 8, "gpu_temp_pause": 87, "gpu_temp_resume": 80},
    },
    "qat": {
        "enabled": True,
        "start_step": 0,                  # 0 = from the first step
        # Weight formats (mercurius.quant.weights.FORMATS): q4_0 (int4, groups of 32, fp16 scale; llama.cpp Q4_0,
        # compressed-tensors int4) | iq4_nl (llama.cpp IQ4_NL levels, groups of 32) | q8_0 | nf4 (legacy) | bf16.
        "weights": "q4_0",
        "gates": "q8_0",                  # GDN-2 decay / erase / write projections
        "embedding": "q4_0",              # tied embedding / lm_head (quantized once, frozen)
        "lm_head": "q8_0",                # only if untied
        "small": "bf16",                  # k_rms, per-head maps, norms, conv, decay vectors: kept
        # KV cache format (mercurius.quant.kv): foveakv | turboquant | int | none
        "kv": {"format": "foveakv", "bits": 4, "head_dim": 128, "block": 128, "gain_bits": 8, "shrink": False,
               "group": 32, "seed": 0},
    },
    "export": {
        "weights": "qat",                 # qat = the QAT formats above, packed; bf16 = full precision
        "dir": "{out_dir}/export",
    },
}


class Cfg(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k) from None
        return v

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


# subtrees whose keys are free-form (user-supplied mappings), not schema keys
_FREE = {("data", "length_mix"), ("data", "episodes"), ("data", "math"), ("train", "anneal")}


def _merge(base, over, path=()):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if path + (k,) in _FREE or (path and path[0] == "teachers"):
            out[k] = v
            continue
        if k not in out:
            raise KeyError(f"unknown config key {'.'.join(path + (k,))}")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v, path + (k,))
        else:
            out[k] = v
    return out


def _parse_value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        low = s.lower()
        if low == "null":                                 # "none" stays a string (amp: none, kv format: none)
            return None
        if low in ("true", "false"):
            return low == "true"
        return s


def _set_path(d, dotted, value):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[int(k)] if isinstance(cur, list) else cur.setdefault(k, {})
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def load_config(path=None, overrides=()):
    raw = {}
    if path:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for o in overrides:
        k, _, v = o.partition("=")
        _set_path(raw, k.strip(), _parse_value(v.strip()))
    cfg = _merge(DEFAULTS, raw)
    cfg["teachers"] = [{**DEFAULTS["teacher_defaults"], **t} for t in cfg["teachers"]]
    for t in cfg["teachers"]:
        unknown = set(t) - set(DEFAULTS["teacher_defaults"])
        if unknown:
            raise KeyError(f"unknown teacher keys {sorted(unknown)}")
    cfg["run"]["out_dir"] = cfg["run"]["out_dir"].format(name=cfg["run"]["name"])
    cfg["export"]["dir"] = cfg["export"]["dir"].format(out_dir=cfg["run"]["out_dir"])
    return validate(_wrap(cfg))


def validate(cfg):
    if not cfg.student.model:
        raise ValueError("student.model is required")
    for t in cfg.teachers:
        if t.kind not in ("hf", "server"):
            raise ValueError(f"teacher kind {t.kind!r}: expected hf or server")
        if t.kind == "server" and not t.url:
            raise ValueError("server teacher needs url")
        if t.kind == "server" and not (t.gguf or t.head):
            raise ValueError("server teacher needs gguf (to read its output head) or head")
        if t.kind == "hf" and not t.model:
            raise ValueError("hf teacher needs model (a path, a hub id, or 'student')")
    if cfg.teacher_combine not in ("loss", "mixture"):
        raise ValueError("teacher_combine: loss | mixture")
    m = cfg.convert.mla
    if m.enabled and m.budget % m.rank_multiple:
        raise ValueError(f"mla.budget {m.budget} is not a multiple of rank_multiple {m.rank_multiple}")
    if cfg.train.amp not in ("none", "bf16", "fp16"):
        raise ValueError("train.amp: none | bf16 | fp16")
    if cfg.train.divergence not in ("forward", "reverse", "js", "jeffreys"):
        raise ValueError(f"train.divergence {cfg.train.divergence!r}")
    taid = cfg.train.taid
    if taid.enabled and cfg.train.divergence == "js":
        raise ValueError("TAID is not derived for js; use forward or reverse")
    if taid.enabled and taid.space == "auto":
        if cfg.train.divergence == "jeffreys":
            raise ValueError("TAID with jeffreys: name train.taid.space (logit or prob) explicitly")
        taid.space = {"forward": "logit", "reverse": "prob"}[cfg.train.divergence]
    lm = cfg.data.length_mix
    if lm:
        cfg.data.length_mix = {int(k): float(v) for k, v in lm.items()}
        if abs(sum(cfg.data.length_mix.values()) - 1.0) > 1e-3:
            raise ValueError("data.length_mix probabilities must sum to 1")
    q = cfg.qat
    from mercurius.quant.weights import FORMATS
    for k in ("weights", "gates", "embedding", "lm_head", "small"):
        if q[k] not in FORMATS:
            raise ValueError(f"qat.{k}: {q[k]!r} is not one of {sorted(FORMATS)}")
    if q.kv.format not in ("foveakv", "turboquant", "int", "none"):
        raise ValueError(f"qat.kv.format {q.kv.format!r}")
    if q.kv.format == "foveakv" and m.enabled and m.rank_multiple % q.kv.head_dim:
        raise ValueError(f"FoveaKV splits the latent into {q.kv.head_dim}-wide heads: mla.rank_multiple "
                         f"({m.rank_multiple}) must be a multiple of qat.kv.head_dim")
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1, default=str)
