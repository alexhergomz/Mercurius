"""Load a checkpoint straight into NF4, one tensor at a time.

transformers' on-the-fly bitsandbytes path (from_pretrained with a
BitsAndBytesConfig) consumed more than 39 GiB of system memory loading the
27B teacher before it was aborted, and in the trainer the kernel OOM-killed
it. On this board that memory is the same unified pool the GPU uses, shared
with vLLM and a speech server.

Here the model is built on the meta device, and every tensor is read from its
safetensors shard, moved to the GPU and, if it is a Linear weight, quantized
there, before the next one is read. Peak memory is the finished NF4 model plus
ONE bf16 matrix (the largest is 17408 x 5120, 170 MiB), by construction.

Quantization matches transformers' bitsandbytes defaults as used by the
trainer (nf4, blocksize 64, double quantization, bf16 compute, lm_head kept
bf16); experiments/test_stream_nf4.py checks the logits against
from_pretrained on a model small enough for both to fit.
"""
import json
import os

import bitsandbytes as bnb
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def _text_config(cfg):
    return getattr(cfg, "text_config", cfg)


def _resolve(model, name):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


@torch.no_grad()
def load_nf4(path, skip=("lm_head",), device="cuda", compute_dtype=torch.bfloat16,
             keep_fp32=True, verbose=True):
    """keep_fp32: tensors the checkpoint stores in fp32 (A_log and the GDN norm
    gains) stay fp32. from_pretrained(dtype=bf16) rounds them to bf16, which
    moves the 4B's logits by relL2 2.4e-2 -- the only difference between the
    two loaders; the NF4 weights themselves are bit-identical. Pass False to
    reproduce from_pretrained exactly."""
    cfg = AutoConfig.from_pretrained(path)
    tcfg = _text_config(cfg)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tcfg, dtype=torch.bfloat16)
    model.eval()

    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        wmap = json.load(open(idx))["weight_map"]
        shards = sorted(set(wmap.values()))
    else:
        shards = ["model.safetensors"]

    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    done, nq = set(), 0
    for sh in shards:
        with safe_open(os.path.join(path, sh), framework="pt", device="cpu") as f:
            for key in f.keys():
                name = key.replace("model.language_model.", "model.")
                if name not in own:
                    continue          # vision tower, MTP head
                parent, leaf = _resolve(model, name)
                t = f.get_tensor(key)
                if (isinstance(parent, nn.Linear) and leaf == "weight"
                        and not any(s in name for s in skip)):
                    lin = parent
                    q = bnb.nn.Linear4bit(
                        lin.in_features, lin.out_features, bias=lin.bias is not None,
                        compute_dtype=compute_dtype, quant_type="nf4",
                        compress_statistics=True, device="cpu")
                    q.weight = bnb.nn.Params4bit(
                        t.to(torch.bfloat16), requires_grad=False,
                        quant_type="nf4", compress_statistics=True, blocksize=64)
                    q = q.to(device)
                    gp, gl = _resolve(model, name.rsplit(".", 1)[0])
                    setattr(gp, gl, q)
                    nq += 1
                else:
                    dt = t.dtype if not t.is_floating_point() else (
                        torch.float32 if keep_fp32 and t.dtype == torch.float32
                        else torch.bfloat16)
                    new = t.to(device=device, dtype=dt)
                    if leaf in parent._parameters:
                        parent._parameters[leaf] = nn.Parameter(new, requires_grad=False)
                    else:
                        parent._buffers[leaf] = new
                done.add(name)
                del t

    # tied output head: point it at the loaded embedding
    if getattr(tcfg, "tie_word_embeddings", False) or getattr(cfg, "tie_word_embeddings", False):
        model.lm_head.weight = model.get_input_embeddings().weight
        done.add("lm_head.weight")

    # Non-persistent buffers (rotary inv_freq) are not in the checkpoint and
    # were created on meta: rebuild every module that holds one on the device.
    for mname, mod in list(model.named_modules()):
        metab = [b for b, v in mod._buffers.items() if v is not None and v.is_meta]
        if not metab:
            continue
        if "rotary" in mname:
            parent, leaf = _resolve(model, mname)
            setattr(parent, leaf, type(mod)(config=tcfg, device=device))
        else:
            raise RuntimeError(f"unrestorable meta buffer(s) {metab} in {mname}")

    left = [n for n, p in model.named_parameters() if p.is_meta]
    if left:
        raise RuntimeError(f"{len(left)} parameters never loaded, e.g. {left[:4]}")
    for p in model.parameters():
        p.requires_grad_(False)
    if verbose:
        print(f"  streamed {path.rstrip('/').split('/')[-1]}: {nq} Linear -> NF4, "
              f"{len(done) - nq} other tensors; "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)
    return model
