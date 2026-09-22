"""Length extrapolation: perplexity by POSITION on long books, one forward each.

A single perplexity at a fixed length hides what extrapolation is about. Here
each book is run once at the full length and the per-token NLL is averaged in
position buckets (0-2k, 2-4k, ... 128-256k). A model that keeps using context
has NLL that falls or holds as position grows; one that breaks past what it was
trained on shows NLL rising in the buckets beyond that point. The student was
distilled only at 8192, while the original was pretrained to 262k with RoPE, so
this is exactly where NoPE + GDN-2 + latent KV either generalise or do not.

Books come from PG-19's test split (Rae et al. 2019), held out from every
corpus used here, and each is used from its start, whole, never concatenated:
a stitched document would put unrelated text at the long positions and measure
nothing about long-range use.

Memory: the lm_head is applied in chunks to the final hidden states, never to
the full sequence (248,320 x 262k logits would be 130 GB).

    python -m mercurius.eval.longppl --arms orig=ORIGINAL_NF4 s8192=ckpt/... \\
        --max-len 262144 --books 4 --quantize
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mercurius import guard
from mercurius.eval.retrieval_ab import build, build_original, build_original_nf4
from mercurius.paths import CACHE_DIR, DATA_DIR, LOGS_DIR, STAGE_AB
from mercurius.surgery.norm_fusion import get_trunk

PG19 = DATA_DIR / "pg19_test_long.jsonl"


def buckets(max_len):
    edges, e = [0, 2048], 2048
    while e < max_len:
        e *= 2
        edges.append(min(e, max_len))
    return list(zip(edges[:-1], edges[1:]))


def fetch_books(tok, min_tokens, n, out=PG19):
    """First n PG-19 test books with at least min_tokens tokens, cached."""
    if out.exists():
        rows = [json.loads(l) for l in open(out)]
        rows = [r for r in rows if r["n_tokens"] >= min_tokens]
        if len(rows) >= n:
            return rows[:n]
    from datasets import load_dataset
    # deepmind/pg19 needs a loading script, which current datasets refuses;
    # emozilla/pg19-test is a parquet mirror of the same test split.
    ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    rows = []
    for rec in ds:
        t = rec["text"]
        if len(t) < min_tokens * 3:          # cheap pre-filter, ~4 chars/token
            continue
        k = len(tok(t, add_special_tokens=False).input_ids)
        if k >= min_tokens:
            rows.append({"title": rec.get("short_book_title", ""), "n_tokens": k,
                         "text": t})
            print(f"  book {len(rows)}: {rows[-1]['title'][:40]!r} {k:,} tokens",
                  flush=True)
        if len(rows) >= n:
            break
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


@torch.no_grad()
def position_nll(model, ids, chunk=4096):
    """Per-position NLL for one sequence, from a single forward."""
    x = ids.unsqueeze(0).cuda()
    h = get_trunk(model)(input_ids=x, use_cache=False).last_hidden_state[0]
    W = model.get_output_embeddings().weight
    out = torch.empty(x.shape[1] - 1, dtype=torch.float32, device="cuda")
    for i in range(0, x.shape[1] - 1, chunk):
        j = min(i + chunk, x.shape[1] - 1)
        out[i:j] = F.cross_entropy((h[i:j] @ W.T).float(), x[0, i + 1:j + 1],
                                   reduction="none")
    del h
    return out.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="tag=path; path ORIGINAL (bf16) or ORIGINAL_NF4 for baselines")
    ap.add_argument("--max-len", type=int, default=131072)
    ap.add_argument("--books", type=int, default=4)
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--covs", default=str(CACHE_DIR / "kv_covs_4b.pt"))
    ap.add_argument("--mla-groups", default=None,
                    help="grouped-latent spec the checkpoint was trained with")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--mem-cap-gb", type=float, default=90.0)
    ap.add_argument("--out", default=str(LOGS_DIR / "longppl.json"))
    a = ap.parse_args()

    guard.cap_cuda_memory(a.mem_cap_gb)
    pacer = guard.ThermalPacer(84.0, 80.0, 90.0, 85.0)
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    books = fetch_books(tok, a.max_len + 1, a.books)
    seqs = [torch.tensor(tok(b["text"], add_special_tokens=False).input_ids[:a.max_len + 1])
            for b in books]
    B = buckets(a.max_len)
    print(f"{len(seqs)} books x {a.max_len:,} tokens; buckets {B}", flush=True)

    results = {}
    for spec in a.arms:
        tag, path = spec.split("=", 1)
        m = (build_original() if path == "ORIGINAL"
             else build_original_nf4() if path == "ORIGINAL_NF4"
             else build(path, a.dc, a.covs, quantize=a.quantize,
                        groups=a.mla_groups))
        pacer.attach(m)
        per_book = []
        for bi, ids in enumerate(seqs):
            t0 = time.time()
            try:
                nll = position_nll(m, ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {tag} book {bi}: OOM at {len(ids):,}", flush=True)
                continue
            row = {f"{lo}-{hi}": nll[lo:hi].mean().item() for lo, hi in B
                   if hi <= len(nll) + 1}
            per_book.append(row)
            print(f"  {tag:<10} book {bi} ({time.time() - t0:5.0f}s) " +
                  " ".join(f"{k}:{math.exp(v):.2f}" for k, v in row.items()),
                  flush=True)
            torch.cuda.empty_cache()
        pacer.detach()
        del m
        torch.cuda.empty_cache()
        keys = list(per_book[0]) if per_book else []
        results[tag] = {k: sum(r[k] for r in per_book) / len(per_book) for k in keys}
        json.dump({"max_len": a.max_len, "books": [b["title"] for b in books],
                   "nll": results}, open(a.out, "w"), indent=1)

    print("\nperplexity by position bucket (mean NLL over books, exponentiated)")
    keys = [f"{lo}-{hi}" for lo, hi in B]
    print(f"{'bucket':>16}" + "".join(f"{t[:12]:>13}" for t in results))
    for k in keys:
        print(f"{k:>16}" + "".join(
            f"{math.exp(results[t][k]):>13.3f}" if k in results[t] else f"{'--':>13}"
            for t in results))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
