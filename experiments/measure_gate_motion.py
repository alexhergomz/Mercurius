"""How far do the KDA gates ACTUALLY move? Gradient norms are the wrong proxy.

AdamW is per-parameter scale-invariant -- the update is lr * m/(sqrt(v)+eps),
roughly lr*sign(g) for consistent gradients -- so a smaller raw gradient does
not imply a smaller step. The question "will KDA refit?" has to be answered by
measuring parameter movement, not gradient magnitude.

Runs a short KL training and reports, per step budget:
  * how far A_log moved from its tiled init
  * whether the CHANNELS DIVERGED (the point of the lift) or moved together
  * the resulting alpha distribution vs the trained-KDA target (median ~0.47)

Channel divergence is the real test. A_log can move a long way while all 128
channels move identically -- that is just a rescaled GDN, and buys nothing.
"""
import sys, time, argparse, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora, freeze_base, trainable_parameters
import bitsandbytes as bnb
from mercurius.paths import BASE_MODEL, FINEWEB, STAGE_AB

CKPT = str(STAGE_AB)
ORIG = str(BASE_MODEL)
DATA = str(FINEWEB)

LORA = [("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
        ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),
        ("mlp.gate_proj", 16), ("mlp.up_proj", 16), ("mlp.down_proj", 16),
        ("lm_head", 0), ("embed_tokens", 0)]


def gate_stats(model):
    """alpha = exp(-exp(A_log) * softplus(.)); report the A_log-driven part."""
    rows = []
    for l in get_trunk(model).layers:
        la = getattr(l, "linear_attn", None)
        if la is None:
            continue
        rows.append(la.A_log.detach().float().clone())
    A = torch.stack(rows)                      # (layers, heads, channels)
    alpha = (-A.exp()).exp()                   # decay at unit softplus
    within = A.std(dim=-1).mean().item()       # spread ACROSS channels in a head
    across = A.mean(dim=-1).std().item()       # spread across heads
    return {"A_mean": A.mean().item(), "within_head_channel_std": within,
            "across_head_std": across, "alpha_median": alpha.median().item(),
            "alpha_min": alpha.min().item(), "alpha_max": alpha.max().item()}


def show(tag, s):
    print(f"  {tag:<22} A_log mean {s['A_mean']:+.4f} | "
          f"channel spread {s['within_head_channel_std']:.5f} | "
          f"alpha med {s['alpha_median']:.4f} "
          f"[{s['alpha_min']:.4f}, {s['alpha_max']:.4f}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gate-lr-mult", type=float, default=1.0)
    ap.add_argument("--lm-weight", type=float, default=0.0)
    ap.add_argument("--seed-decay", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]

    teacher = AutoModelForCausalLM.from_pretrained(
        ORIG, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load_kda_model(CKPT, dtype=torch.bfloat16)
    if a.seed_decay:
        for l in get_trunk(student).layers:
            if hasattr(l, "linear_attn"):
                l.linear_attn.seed_decay_from_rope()
    install_rope_dial(student, 0, "global")     # NoPE
    inject_lora(student, LORA, verbose=False)
    freeze_base(student)

    init = gate_stats(student)
    A0 = torch.stack([l.linear_attn.A_log.detach().float().clone()
                      for l in get_trunk(student).layers
                      if hasattr(l, "linear_attn")])

    print(f"\nlr {a.lr:.1e}  gate-mult {a.gate_lr_mult:g}  "
          f"lm-weight {a.lm_weight}  seeded {a.seed_decay}")
    show("init", init)

    params = trainable_parameters(student)
    # No gate LR multiplier: AdamW is per-parameter scale-invariant, so the
    # smaller raw gradient on the gates does not mean smaller steps. Measurement
    # confirmed 98.6% of gate motion is channel-differential -- direction is
    # fine, magnitude is the issue, and that is an INITIALIZATION problem.
    opt = bnb.optim.AdamW8bit(params, lr=a.lr, betas=(0.9, 0.95))

    g = torch.Generator().manual_seed(0)
    student.train()
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        i = int(torch.randint(0, len(ids) - a.seq - 1, (1,), generator=g))
        x = ids[i:i + a.seq].unsqueeze(0).cuda()
        with torch.no_grad():
            tl = teacher(input_ids=x).logits[0]
        sl = student(input_ids=x).logits[0]
        loss = 0.0
        for s0 in range(0, sl.shape[0], 512):
            s_c, t_c = sl[s0:s0+512].float(), tl[s0:s0+512].float()
            loss = loss + (F.softmax(t_c, -1) *
                           (F.log_softmax(t_c, -1) - F.log_softmax(s_c, -1))
                           ).sum(-1).sum()
        loss = loss / sl.shape[0]
        if a.lm_weight > 0:
            ce = F.cross_entropy(sl[:-1].float(), x[0, 1:])
            loss = loss + a.lm_weight * ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        del tl, sl
        if step % 40 == 0:
            A = torch.stack([l.linear_attn.A_log.detach().float()
                             for l in get_trunk(student).layers
                             if hasattr(l, "linear_attn")])
            moved = (A - A0).abs().mean().item()
            show(f"step {step}", gate_stats(student))
            print(f"    |A_log - init| mean {moved:.6f}   "
                  f"channel-spread change "
                  f"{gate_stats(student)['within_head_channel_std'] - init['within_head_channel_std']:+.6f}",
                  flush=True)
        torch.cuda.empty_cache()

    fin = gate_stats(student)
    A = torch.stack([l.linear_attn.A_log.detach().float()
                     for l in get_trunk(student).layers
                     if hasattr(l, "linear_attn")])
    print(f"\n  tokens seen: {a.steps * a.seq / 1e6:.2f} M   "
          f"({time.perf_counter()-t0:.0f}s)")
    print(f"  |A_log - init| mean {(A - A0).abs().mean().item():.6f}  "
          f"max {(A - A0).abs().max().item():.6f}")
    print(f"  channel spread {init['within_head_channel_std']:.5f} -> "
          f"{fin['within_head_channel_std']:.5f}  "
          f"({'DIVERGING' if fin['within_head_channel_std'] > init['within_head_channel_std'] * 1.05 else 'NOT diverging'})")
    print(f"  alpha median {init['alpha_median']:.4f} -> {fin['alpha_median']:.4f}"
          f"   (trained-KDA target ~0.47)")


if __name__ == "__main__":
    raise SystemExit(main())
