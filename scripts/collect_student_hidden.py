"""Student final hidden states on the SAME token positions as the teacher cache.

Pairing must be exact: P is fitted from (h_s, h_t) at matching positions, so a
one-token drift would fit noise. The teacher cache stores the token ids it
actually kept, and llama.cpp's tokenizer was verified identical to the HF one
(1007/1007 ids on a calib_mix sample), so the ids are replayed here directly
rather than re-tokenising text.

The state taken is the trunk's last_hidden_state -- after the final norm, before
lm_head -- which is the same object llama.cpp returns for the teacher and the
same one `_chunk_div_terms` consumes today.
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from mercurius.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-cache", default=str(ROOT / "cache/h_teacher_calib.npz"))
    ap.add_argument("--ckpt", default="ckpt/adapters-masked150-best.pt")
    ap.add_argument("--out", default=str(ROOT / "cache/h_student_calib.npz"))
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--covs", default=str(ROOT / "cache/kv_covs_4b_mix.pt"))
    ap.add_argument("--groups", default=str(ROOT / "cache/mla_groups_retr_4096_mix.json"))
    ap.add_argument("--dc", type=int, default=512)
    a = ap.parse_args()

    from mercurius import guard
    from mercurius.eval.retrieval_ab import build, build_original_nf4
    from mercurius.recovery.train import get_trunk
    guard.cap_cuda_memory(60)

    ids = np.load(a.teacher_cache)["ids"]
    print(f"{len(ids):,} positions from {os.path.basename(a.teacher_cache)}", flush=True)
    model = (build_original_nf4() if a.ckpt in ("original", "base", "none")
             else build(a.ckpt, a.dc, a.covs, quantize=True, groups=a.groups))
    model.eval()
    trunk = get_trunk(model)

    out = []
    with torch.no_grad():
        for i in range(0, len(ids), a.chunk):
            x = torch.tensor(ids[i:i + a.chunk], dtype=torch.long).unsqueeze(0).cuda()
            h = trunk(input_ids=x).last_hidden_state[0]
            out.append(h.float().cpu().numpy())
            if (i // a.chunk + 1) % 10 == 0:
                print(f"  {i + x.shape[1]:,}/{len(ids):,}", flush=True)
    H = np.concatenate(out, 0)
    print(f"collected {H.shape} | norms mean {np.linalg.norm(H, axis=-1).mean():.1f}")
    np.savez(a.out, h=H.astype(np.float16), ids=ids)
    print(f"-> {a.out} ({os.path.getsize(a.out)/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
