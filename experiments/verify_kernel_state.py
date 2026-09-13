"""Is the KDA kernel's final_state a valid initial_state for the next segment?

Isolates the question from the model: run one sequence in one call, versus the
same sequence split in two with the state handed across. If the state
convention is right these agree to kernel noise. If they do not, the model-level
state passing cannot possibly work.
"""
import torch
from fla.ops import chunk_kda


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


def main():
    torch.manual_seed(0)
    B, T, H, D = 1, 512, 4, 128
    half = T // 2
    dev, dt = "cuda", torch.bfloat16

    q = torch.randn(B, T, H, D, device=dev, dtype=dt)
    k = torch.randn(B, T, H, D, device=dev, dtype=dt)
    v = torch.randn(B, T, H, D, device=dev, dtype=dt)
    beta = torch.rand(B, T, H, device=dev, dtype=dt) * 0.5 + 0.25
    g = (-torch.rand(B, T, H, D, device=dev) * 0.05).float()

    kw = dict(use_qk_l2norm_in_kernel=True)

    # one shot
    full, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta,
                        output_final_state=False, **kw)

    # split, carrying state
    o1, s1 = chunk_kda(q=q[:, :half], k=k[:, :half], v=v[:, :half],
                       g=g[:, :half], beta=beta[:, :half],
                       output_final_state=True, **kw)
    print(f"  final_state shape: {tuple(s1.shape)}  dtype {s1.dtype}")

    for label, st in (("as-returned", s1),
                      ("transposed(-1,-2)", s1.transpose(-1, -2).contiguous())):
        try:
            o2, _ = chunk_kda(q=q[:, half:], k=k[:, half:], v=v[:, half:],
                              g=g[:, half:], beta=beta[:, half:],
                              initial_state=st, output_final_state=False, **kw)
            d = rel(full[:, half:], o2)
            print(f"  second half, initial_state {label:<20} rel {d:.5f}"
                  f"   {'MATCH' if d < 0.05 else 'MISMATCH'}")
        except Exception as e:
            print(f"  second half, initial_state {label:<20} ERROR {str(e)[:90]}")

    # control: no state at all
    o0, _ = chunk_kda(q=q[:, half:], k=k[:, half:], v=v[:, half:],
                      g=g[:, half:], beta=beta[:, half:],
                      output_final_state=False, **kw)
    print(f"  second half, NO initial_state{'':<12} rel {rel(full[:, half:], o0):.5f}"
          "   (baseline: how wrong you are with no history)")

    # does state_v_first change the convention?
    o1b, s1b = chunk_kda(q=q[:, :half], k=k[:, :half], v=v[:, :half],
                         g=g[:, :half], beta=beta[:, :half],
                         output_final_state=True, state_v_first=True, **kw)
    o2b, _ = chunk_kda(q=q[:, half:], k=k[:, half:], v=v[:, half:],
                       g=g[:, half:], beta=beta[:, half:],
                       initial_state=s1b, output_final_state=False,
                       state_v_first=True, **kw)
    print(f"  with state_v_first=True{'':<18} rel {rel(full[:, half:], o2b):.5f}")


if __name__ == "__main__":
    main()
