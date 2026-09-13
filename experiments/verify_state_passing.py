"""Does state passing actually extend effective context, or just set a flag?

Three checks:

  1. Carrying state CHANGES the output. If the kernel silently ignored
     initial_state, segment B alone and B-after-A would be identical.

  2. B-after-A approximates the true long-context result. Compare against
     running [A;B] as one full sequence and reading the B positions. If state
     passing works, the two should be close -- that is the whole claim.

  3. The approximation holds as the carried history grows (2, 4, 8 segments).

Check 2 is the load-bearing one: it is the difference between "the state is
plumbed through" and "the model genuinely sees the earlier context."
"""
import sys, torch
from transformers import AutoTokenizer
from mercurius.models.kda import (load_kda_model, enable_state_passing, reset_state,
                       promote_state)
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.paths import STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
SEG = 512


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


@torch.no_grad()
def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]

    model = load_kda_model(CKPT, dtype=torch.bfloat16)
    install_rope_dial(model, 0, "global")        # NoPE, as in the training run
    model.eval()

    for n_hist in (1, 3, 7):
        total = (n_hist + 1) * SEG
        seq = ids[:total]
        segs = [seq[i * SEG:(i + 1) * SEG].unsqueeze(0).cuda()
                for i in range(n_hist + 1)]
        last = segs[-1]

        # --- reference: the whole thing as one sequence ---
        enable_state_passing(model, False)
        full = model(input_ids=seq.unsqueeze(0).cuda()).logits[0, -SEG:]

        # --- cold: final segment with NO history ---
        cold = model(input_ids=last).logits[0]

        # --- state passing: walk the history, then the final segment ---
        enable_state_passing(model, True)
        reset_state(model)
        for s in segs[:-1]:
            model(input_ids=s)
            promote_state(model)
        carried = model(input_ids=last).logits[0]
        enable_state_passing(model, False)

        d_cold = rel(full, cold)
        d_carry = rel(full, carried)
        print(f"  history {n_hist * SEG:>5} tok | "
              f"cold vs full {d_cold:.4f} | carried vs full {d_carry:.4f} | "
              f"closes {(1 - d_carry / max(d_cold, 1e-9)) * 100:5.1f}% of the gap",
              flush=True)
        torch.cuda.empty_cache()

    print("\n  check 1 (state changes output): "
          f"{'PASS' if d_carry != d_cold else 'FAIL - state ignored'}")
    print("  check 2 (carried approximates full context): "
          f"{'PASS' if d_carry < d_cold * 0.6 else 'WEAK - little benefit'}")


if __name__ == "__main__":
    main()
