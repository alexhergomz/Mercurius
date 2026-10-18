"""Build D13's teacher hidden-state cache using llama.cpp, from the GGUF.

D5 routes the training-time teacher to 35B-A3B and records the blocker: Qwen3.5
MoE stores experts as fused 3-D tensors that `bnb.Linear4bit` cannot wrap. That
blocker is real, but it gates in-process teacher LOGITS -- and D13 already
removed the need for those by caching the final hidden state instead. The cache
can therefore be built today, from the Q4_K_M GGUF already on disk, with no
loader and no download.

VERIFIED, not assumed. llama.cpp's per-token embeddings are the post-output_norm,
pre-lm_head state: decoding W_t h_t (head dequantised from the same GGUF with
gguf.quants.dequantize) gives coherent continuations -- ' fibonacci'->'(n',
'):'->newline, ' else'->' fibonacci', max prob 0.92-0.95 on syntax, 6.14 nats at
position 0 where there is no context. Two flags are load-bearing:

    --pooling none          per token, not one vector per sequence
    --embd-normalize -1     RAW. The default L2-normalises to unit length, which
                            silently destroys the scale the loss depends on.
                            Observed norms are ~114; if they come back ~1.0 the
                            flag did not take and every cached value is wrong.

Storage follows D13: int8 per channel, measured there at 0.05% reverse-KL error
against bf16, where PCA at any rank failed catastrophically off-domain because
the final hidden state is near full rank by construction.

    llama-server -m <gguf> --embeddings --pooling none --embd-normalize -1 \
        -ngl 99 -c 4096 -b 4096 --port 8077
    python scripts/collect_teacher_hidden.py --input data/calib_mix.txt \
        --out cache/h_teacher_calib.npz --max-tokens 100000
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, "src")
from mercurius.paths import STAGE_AB


def embed(url, text, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps({"input": text}).encode(),
        headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    if isinstance(d, dict):
        d = d.get("data", [d])
    return np.asarray(d[0]["embedding"], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8077/embeddings")
    ap.add_argument("--chunk-tokens", type=int, default=1024,
                    help="tokens per request; must be <= the server's -c")
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--int8", action="store_true", default=True,
                    help="store int8 per channel (D13). --no-int8 for raw fp16")
    ap.add_argument("--no-int8", dest="int8", action="store_false")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    ids = tok(open(a.input, encoding="utf-8", errors="replace").read(),
              add_special_tokens=False).input_ids
    ids = ids[:a.max_tokens]
    print(f"{len(ids):,} tokens from {a.input}", flush=True)

    chunks, H, kept = [], [], []
    for i in range(0, len(ids), a.chunk_tokens):
        chunks.append(ids[i:i + a.chunk_tokens])
    t0 = time.time()
    for n, c in enumerate(chunks):
        h = embed(a.url, tok.decode(c))
        # the server retokenises the decoded text, so its token count can differ
        # by a few; keep the positions we can actually pair with an id
        m = min(len(h), len(c))
        H.append(h[:m]); kept.extend(c[:m])
        if (n + 1) % 10 == 0 or n + 1 == len(chunks):
            done = sum(len(x) for x in H)
            print(f"  {n+1}/{len(chunks)} chunks, {done:,} positions, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    H = np.concatenate(H, 0)
    ids_arr = np.asarray(kept, dtype=np.int32)
    print(f"collected {H.shape} | norms mean {np.linalg.norm(H,axis=-1).mean():.1f}")
    if abs(np.linalg.norm(H, axis=-1).mean() - 1.0) < 0.05:
        sys.exit("norms are ~1.0: the server L2-normalised. Re-run with "
                 "--embd-normalize -1 or every cached value is wrong.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.int8:
        scale = np.abs(H).max(0) / 127.0                 # per channel, D13
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(H / scale), -127, 127).astype(np.int8)
        err = np.abs(q.astype(np.float32) * scale - H).max()
        np.savez(a.out, h_int8=q, scale=scale.astype(np.float32), ids=ids_arr,
                 dtype="int8_per_channel")
        print(f"int8 per channel, max abs error {err:.5f} "
              f"({100*err/np.abs(H).max():.3f}% of absmax)")
    else:
        np.savez(a.out, h=H.astype(np.float16), ids=ids_arr, dtype="fp16")
    print(f"-> {a.out}  ({os.path.getsize(a.out)/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
