"""Which parameters actually receive gradient under each objective?

Claim under test: "KL only constrains the attention layers, so the KDA layers
are free to do what they want."

  train_recovery.py  -- KL between FINAL LOGITS. One end-to-end objective, so
                        gradient flows through every layer including KDA.
  train_transfer.py  -- MSE on 6 attention layers run standalone on teacher
                        inputs. The KDA layers never execute in the loss path.

Reports gradient norms by parameter group so the answer is measured, not argued.
"""
import sys, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora, freeze_base, trainable_parameters
from mercurius.paths import BASE_MODEL, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
ORIG = str(BASE_MODEL)

KL_LORA = [("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
           ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),
           ("linear_attn.out_proj", 16), ("linear_attn.in_proj_qkv", 16),
           ("mlp.gate_proj", 16), ("mlp.up_proj", 16), ("mlp.down_proj", 16),
           ("lm_head", 0), ("embed_tokens", 0)]
ATTN_LORA = [("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
             ("self_attn.v_proj", 32), ("self_attn.o_proj", 32)]


def group_of(name):
    if "a_lora_" in name or name.endswith("A_log") or name.endswith("dt_bias"):
        return "KDA gate (stage B capacity)"
    if "linear_attn" in name:
        return "KDA other (proj LoRA)"
    if "self_attn" in name:
        return "attention LoRA"
    if "mlp" in name:
        return "MLP LoRA"
    return "other"


def audit(tag, model, loss):
    loss.backward()
    agg = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = group_of(n)
        norm = 0.0 if p.grad is None else p.grad.float().norm().item()
        cnt, tot, nz = agg.get(g, (0, 0.0, 0))
        agg[g] = (cnt + 1, tot + norm, nz + (1 if norm > 0 else 0))
    print(f"\n=== {tag} ===")
    for g, (cnt, tot, nz) in sorted(agg.items()):
        state = "TRAINS" if nz else "NO GRADIENT"
        print(f"  {g:<32} {nz:>3}/{cnt:<3} tensors with grad  "
              f"sum|grad| {tot:10.5f}   {state}")
    model.zero_grad(set_to_none=True)


def build(lora_rules, dial="nope"):
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    keep, pol = {"nope": (0, "global"), "c0": (32, "local")}[dial]
    install_rope_dial(m, keep, pol)
    inject_lora(m, lora_rules, verbose=False)
    freeze_base(m)
    return m


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(str(WIKITEXT)).read(),
              return_tensors="pt").input_ids[0]
    x = ids[:512].unsqueeze(0).cuda()

    teacher = AutoModelForCausalLM.from_pretrained(
        ORIG, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---------- objective 1: end-to-end KL on final logits ----------
    m = build(KL_LORA)
    with torch.no_grad():
        t_log = teacher(input_ids=x).logits[0]
    s_log = m(input_ids=x).logits[0]
    lp_s = F.log_softmax(s_log.float(), -1)
    p_t = F.softmax(t_log.float(), -1)
    kl = (p_t * (F.log_softmax(t_log.float(), -1) - lp_s)).sum(-1).mean()
    audit("objective: end-to-end KL (train_recovery.py)", m, kl)
    del m, s_log; torch.cuda.empty_cache()

    # ---------- objective 2: layer-local attention transfer ----------
    m = build(ATTN_LORA)
    idxs = [i for i, l in enumerate(get_trunk(m).layers) if hasattr(l, "self_attn")]
    with torch.no_grad():
        th = teacher(input_ids=x, output_hidden_states=True).hidden_states
    pos = torch.arange(512, device=x.device).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = get_trunk(m).rotary_emb(th[0], pos)
    layers = get_trunk(m).layers
    loss = 0.0
    for li in idxs:
        out = layers[li](hidden_states=th[li].detach(),
                         position_embeddings=(cos, sin), attention_mask=None)
        if isinstance(out, tuple):
            out = out[0]
        loss = loss + F.mse_loss(out.float(), th[li + 1].detach().float())
    audit("objective: layer-local attention transfer (train_transfer.py)",
          m, loss / len(idxs))
    del m; torch.cuda.empty_cache()

    # ---------- objective 3: teacher-free LM cross-entropy ----------
    m = build(ATTN_LORA)
    logits = m(input_ids=x).logits[0]
    ce = F.cross_entropy(logits[:-1].float(), x[0, 1:])
    audit("objective: LM cross-entropy, no teacher (--lm-weight)", m, ce)


if __name__ == "__main__":
    main()
