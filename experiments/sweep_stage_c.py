"""Stage C sweep: how much does removing rotary dimensions cost, before training?

Measures the IMMEDIATE damage of the dial at several settings and policies, with
no recovery training. This is the C0..C3 staircase from the plan, and it also
answers which end of the frequency ladder matters.

Two probe lengths, because RoPE removal is a positional intervention: a 48-token
probe barely exercises position, a 512-token one exercises it more. Neither is a
substitute for a RULER-style long-context eval, which is the real G4 gate.

The model here is the Stage A+B checkpoint, so the dial is measured on top of
the converted architecture, not the original.
"""
import sys, torch
from mercurius.harness import Harness, rel
from mercurius.models.kda import load_kda_model
from mercurius.surgery.rope_dial import install_rope_dial, dial_summary
from mercurius.paths import STAGE_AB

CKPT = str(STAGE_AB)

LONG_TEXT = (
    "The Jetson AGX Orin is an embedded platform with unified memory. "
    "Linear attention maintains a fixed-size state matrix rather than a "
    "growing key-value cache, so its cost is linear in sequence length. "
    "The delta rule corrects the state toward the current value, and a gate "
    "controls how quickly old associations decay. When positional encoding is "
    "removed from the full-attention layers, the linear layers must carry the "
    "positional signal themselves, through their decay rates and causal "
    "convolution. Whether that is sufficient at long context is the open "
    "question this project exists to answer. "
) * 12


def sweep(h, model, ref, label):
    print(f"\n--- {label} ---", flush=True)
    base = h.measure("C0  full RoPE (64 dims)", model, ref)
    rows = [("C0", 32, "-", base)]
    for keep, tag in ((16, "C1"), (8, "C1b"), (4, "C2"), (0, "C3")):
        for policy in (("global", "local", "stride") if keep else ("global",)):
            restore = install_rope_dial(model, keep, policy)
            r = h.measure(f"{tag}  {keep*2:>2} dims  {policy:<7}", model, ref)
            rows.append((tag, keep, policy, r))
            restore()
    return rows


def main():
    torch.manual_seed(0)

    print("=== short probe (48 tokens) ===", flush=True)
    h = Harness(CKPT, max_length=48)
    model = load_kda_model(CKPT, dtype=torch.float32)
    print(f"loaded: {dial_summary(model)}", flush=True)
    ref = h.reference(model)
    short = sweep(h, model, ref, "48-token probe")

    print("\n\n=== long probe (512 tokens) ===", flush=True)
    h2 = Harness(CKPT, texts=[LONG_TEXT], max_length=512)
    print(f"probe tokens: {h2.batch['input_ids'].shape}", flush=True)
    ref2 = h2.reference(model)
    long = sweep(h2, model, ref2, "512-token probe")

    print("\n\n=== summary: relL2 vs full RoPE ===")
    print(f"  {'setting':<24} {'48 tok':>12} {'512 tok':>12}")
    for (t, k, p, rs), (_, _, _, rl) in zip(short, long):
        name = f"{t} {k*2:>2}d {p}"
        print(f"  {name:<24} {rs['rel']:>12.3e} {rl['rel']:>12.3e}")

    print("\n  top-1 agreement, 512-token probe:")
    for t, k, p, r in long:
        print(f"    {t} {k*2:>2}d {p:<7} {r['top1']:7.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
