"""Does a checkpoint REPLAY to the model that was trained? Check before benchmarking it.

build() reconstructs an arm from its adapters with strict=False, so a key it fails to
match (a MoL stack, a router, a routing metric) is DROPPED SILENTLY and every benchmark
after it measures a different model. This rebuilds the arm exactly as the benchmarks
will -- same build(), same --quantize (the trainer's NF4 base), same dial, groups and
covariances -- and recomputes the trainer's own eval metric (suite.ce_and_topk on the
held-out WikiText, first n tokens) at 2048 and 8192. It must match the training log's
final eval, or the arm is not benchmarked.

    .venv/bin/python experiments/replay_check.py --arm A=ckpt/adapters-c0-mol1-care-150.pt \
        --groups cache/mla_groups_ungrouped_2048.json --expect 2.0443 2.2608
"""
import argparse
import sys

import torch
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="tag=path")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--covs", default="cache/kv_covs_4b_mix.pt")
    ap.add_argument("--dial", default="c0")
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--expect", type=float, nargs=2, required=True,
                    metavar=("CE2048", "CE8192"), help="final eval CE from the training log")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--qat", action="store_true",
                    help="rebuild as the DEPLOYED 4-bit model (models/qat.py) -- "
                         "required for checkpoints of a --qat run")
    ap.add_argument("--qat-kv-bits", type=int, default=4)
    ap.add_argument("--qat-kv-group", type=int, default=32)
    ap.add_argument("--qat-kv-rot", default="none", choices=["none", "orth"])
    ap.add_argument("--qat-kv-quant", default="int", choices=["int", "tq"],
                    help="KV latent quantizer: int = symmetric int, fp16 scale per "
                         "--qat-kv-group; tq = TurboQuant-MSE (no QJL): random rotation, "
                         "fp16 norm per token, Beta Lloyd-Max codebook (#66)")
    ap.add_argument("--qat-gate-bits", type=int, default=4)
    ap.add_argument("--qat-embed-bits", type=int, default=4)
    ap.add_argument("--merge-eval", action="store_true",
                    help="merge VeRA into bf16 bases for inference (build(merge_eval=True)): "
                         "~2.5x faster at rank 1024, numerics differ only by one bf16 "
                         "rounding of the merged weight -- check with replay_check --merge-eval")
    a = ap.parse_args()
    _qat = (dict(kv_bits=a.qat_kv_bits, kv_group=a.qat_kv_group, kv_rot=a.qat_kv_rot,
                 gate_bits=a.qat_gate_bits, embed_bits=a.qat_embed_bits,
                 kv_quant=a.qat_kv_quant)
            if a.qat else None)

    from mercurius.eval.retrieval_ab import build
    from mercurius.eval.suite import ce_and_topk
    from mercurius.recovery.train import CKPT
    from mercurius.paths import WIKITEXT
    tag, path = a.arm.split("=", 1)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(WIKITEXT).read(), return_tensors="pt").input_ids[0]
    m = build(path, a.dc, a.covs, groups=a.groups, dial=a.dial, quantize=True, qat=_qat,
              merge_eval=a.merge_eval)
    m.eval()
    bad = False
    with torch.no_grad():
        for n, exp in zip((2048, 8192), a.expect):
            ce = float(ce_and_topk(m, ids, n)["ce"])
            d = ce - exp
            ok = abs(d) <= a.tol
            bad |= not ok
            print(f"[replay] {tag} @{n}: CE {ce:.4f} vs trained {exp:.4f}  "
                  f"diff {d:+.4f}  {'OK' if ok else 'MISMATCH'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
