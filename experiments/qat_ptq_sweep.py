"""PTQ cost of the deployed 4-bit format, decomposed, and the KV-latent choice (#64).

Run at the leg boundary on the leg-1 final adapters, BEFORE leg 2 starts with --qat:
rebuilds the trained model exactly as the harnesses do (replay, NF4 base) and
measures the trainer's eval CE (WikiText, @2048 / @8192) for

    trained          as leg 1 left it (bf16 adapters / latents / embedding)
    + W4             every matmul weight NF4 in deployed form (VeRA merged, maps and
                     rotation folded); the 755 M GDN-2 gate weights bf16 / int8 / NF4
                     (0 / 0.80 / 0.39 GB); KV and embedding untouched. The KV and
                     embedding rows keep gates at NF4.
    + KV variants    int4 latent, groups 16/32/64, rotation none/orth
    + embedding      the tied embedding NF4, on top of the best KV variant

so leg 2 starts from a measured choice of --qat-kv-group / --qat-kv-rot, and the
QAT run's first eval (the resumed model, quantized) can be checked against it.

    .venv/bin/python experiments/qat_ptq_sweep.py --adapters ckpt/adapters-c0-long75-step7875.pt \
        --covs cache/mla_seqcal_a860e70514_covs.pt --groups cache/mla_seqcal_a860e70514_groups.json
"""
import argparse
import json

import torch
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--covs", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--dial", default="c0")
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--out", default="logs/qat_ptq_sweep.json")
    ap.add_argument("--only-tq", action="store_true",
                    help="trained, +W4 (gates bf16), then TurboQuant 3/4-bit and the "
                         "int4 g32 reference only")
    a = ap.parse_args()

    from mercurius.eval.retrieval_ab import build
    from mercurius.eval.suite import ce_and_topk
    from mercurius.models.qat import install_qat
    from mercurius.surgery.transmla import LatentKV
    from mercurius.models import qat as Q
    from mercurius.recovery.train import CKPT
    from mercurius.paths import WIKITEXT
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(WIKITEXT).read(), return_tensors="pt").input_ids[0]
    m = build(a.adapters, a.dc, a.covs, groups=a.groups, dial=a.dial, quantize=True).eval()
    rows = {}

    def ev(name):
        with torch.no_grad():
            r = {n: float(ce_and_topk(m, ids, n)["ce"]) for n in (2048, 8192)}
        rows[name] = r
        base = rows.get("trained")
        d = "" if base is None else "  delta " + " / ".join(
            f"{r[n] - base[n]:+.4f}" for n in (2048, 8192))
        print(f"[ptq] {name:28s} CE {r[2048]:.4f} / {r[8192]:.4f}{d}", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)

    ev("trained")
    install_qat(m, kv_bits=16, embed_bits=16, gate_bits=16)
    lat = [mm for mm in m.modules() if isinstance(mm, LatentKV)]
    cfg = lat[0]._qat                       # one dict shared by every module
    if a.only_tq:
        ev("+W4 body (gates bf16)")
        cfg["kv_bits"], cfg["kv_group"] = 4, 32
        ev("+KV int4 g32 rot=none")
        for i, mm in enumerate(lat):
            mm._qat_rot = Q._orth(mm.down.weight.shape[0], 9000 + i)
        cfg["kv_quant"] = "tq"
        for b in (3, 4):
            cfg["kv_bits"] = b
            ev(f"+KV TurboQuant {b}-bit")
        return
    ev("+W4 body (gates bf16)")
    cfg["gate_bits"] = 8
    ev("+W4 body, gates int8")
    cfg["gate_bits"] = 4
    ev("+W4 all (gates NF4)")
    best = None
    for rot in ("none", "orth"):
        for i, mm in enumerate(lat):
            mm._qat_rot = Q._orth(mm.down.weight.shape[0], 9000 + i) if rot == "orth" else None
        for g in (16, 32, 64):
            cfg["kv_bits"], cfg["kv_group"] = 4, g
            name = f"+KV int4 g{g} rot={rot}"
            ev(name)
            # 4.5 bits/value at g32 is the deployed budget; g16 costs 5.0
            if g == 32 and (best is None or rows[name][8192] < rows[best][8192]):
                best = name
    # TurboQuant-MSE (#66): its rotation is part of the quantizer
    for i, mm in enumerate(lat):
        mm._qat_rot = Q._orth(mm.down.weight.shape[0], 9000 + i)
    cfg["kv_quant"] = "tq"
    for b in (3, 4):
        cfg["kv_bits"] = b
        ev(f"+KV TurboQuant {b}-bit")
    cfg["kv_quant"], cfg["kv_bits"] = "int", 4
    rot = best.split("rot=")[1]
    for i, mm in enumerate(lat):
        mm._qat_rot = Q._orth(mm.down.weight.shape[0], 9000 + i) if rot == "orth" else None
    cfg["kv_bits"], cfg["kv_group"] = 4, 32
    install_qat(m, kv_bits=4, kv_group=32, kv_rot=rot, embed_bits=4)
    ev(f"+embedding NF4 (KV g32 {rot})")
    print(f"[ptq] chosen KV rotation at g32: {rot}  -> --qat-kv-rot {rot}", flush=True)


if __name__ == "__main__":
    main()
