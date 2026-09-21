"""Does the trained student actually run Gated DeltaNet-2, not KDA?

Builds the student the way the trainer does (stage A+B, NoPE, --gdn2, VeRA on
the gate projections), then checks:
  1. every linear-attention layer is Qwen3_5GDN2GatedDeltaNet;
  2. in forward + backward only the GDN-2 kernels are called, never KDA's;
  3. the lift is exact at init (logits equal to the KDA model it came from);
  4. the GDN-2 erase/write gates receive gradient.

    python experiments/test_gdn2_active.py
"""
import torch
import fla.ops as fops
import fla.ops.gdn2 as fgdn2

import mercurius.models.kda as kda_mod
import mercurius.models.gdn2 as gdn2_mod
from mercurius.models.kda import load_kda_model, Qwen3_5KDAGatedDeltaNet
from mercurius.models.gdn2 import convert_to_gdn2, Qwen3_5GDN2GatedDeltaNet
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_vera, freeze_base
from mercurius.paths import STAGE_AB

calls = {}


def count(name, fn):
    def wrapped(*a, **k):
        calls[name] = calls.get(name, 0) + 1
        return fn(*a, **k)
    return wrapped


# the modules bind the kernels at import time, so patch their references
for mod in (kda_mod,):
    mod.chunk_kda = count("chunk_kda", mod.chunk_kda)
    mod.fused_recurrent_kda = count("fused_recurrent_kda", mod.fused_recurrent_kda)
gdn2_mod.chunk_gdn2 = count("chunk_gdn2", gdn2_mod.chunk_gdn2)
gdn2_mod.fused_recurrent_gdn2 = count("fused_recurrent_gdn2", gdn2_mod.fused_recurrent_gdn2)

ok = []
torch.manual_seed(0)
m = load_kda_model(str(STAGE_AB), dtype=torch.bfloat16)
install_rope_dial(m, 0, "global")
x = torch.randint(1000, 50000, (1, 1024), device="cuda")
with torch.no_grad():
    ref = m(input_ids=x, use_cache=False).logits.float()
calls.clear()

convert_to_gdn2(m, verbose=False)
kinds = {}
for l in get_trunk(m).layers:
    la = getattr(l, "linear_attn", None)
    if la is not None:
        kinds[type(la).__name__] = kinds.get(type(la).__name__, 0) + 1
ok.append(kinds == {"Qwen3_5GDN2GatedDeltaNet": 24})
print(f"1. linear-attention layer classes: {kinds}  [{'PASS' if ok[-1] else 'FAIL'}]")

with torch.no_grad():
    got = m(input_ids=x, use_cache=False).logits.float()
r = ((got - ref).norm() / ref.norm()).item()
ok.append(r < 1e-2)
print(f"3. lift vs KDA at init: logits relL2 {r:.2e} (bf16, different kernel)  "
      f"[{'PASS' if ok[-1] else 'FAIL'}]")

inject_vera(m, [("linear_attn.in_proj_be", 1), ("linear_attn.in_proj_bw", 1)],
            rank=64, verbose=False)
freeze_base(m, also_train=("vera_d", "vera_b"))
for n, p in m.named_parameters():      # nudge b off zero so gradient reaches d too
    if "vera_b" in n:
        p.data.normal_(0, 1e-3)
calls.clear()
m.train()
out = m(input_ids=x, use_cache=False).logits.float()
out.logsumexp(-1).mean().backward()
fwd_bwd = dict(calls)
ok.append(fwd_bwd.get("chunk_gdn2", 0) == 24 and not any("kda" in k for k in fwd_bwd))
print(f"2. kernel calls in forward+backward: {fwd_bwd}  [{'PASS' if ok[-1] else 'FAIL'}]")
g = {n: p.grad.abs().sum().item() for n, p in m.named_parameters()
     if p.requires_grad and p.grad is not None}
gb = sum(v for n, v in g.items() if "in_proj_be" in n)
gw = sum(v for n, v in g.items() if "in_proj_bw" in n)
ok.append(gb > 0 and gw > 0)
print(f"4. gradient on erase gate {gb:.3e}, write gate {gw:.3e}  "
      f"[{'PASS' if ok[-1] else 'FAIL'}]")
print(f"\n{sum(ok)}/{len(ok)} checks passed")
raise SystemExit(0 if all(ok) else 1)
