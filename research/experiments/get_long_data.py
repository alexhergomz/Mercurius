"""Build a long-document training corpus by filtering FineWeb-Edu while streaming.

Why this exists. The training corpus is tokenized as ONE stream and windows are
drawn at a random offset into it, so an 8192-token window is whatever happens to
be adjacent. Measured on data/fineweb_edu.txt: median document 550 tokens, mean
934, p90 2041, and only 0.8% reach 8192. An 8192 window therefore spans about 15
unrelated documents.

The consequence is that no run in this project has trained on a real long-range
dependency. The longest genuine dependency in the data ends around 2k tokens,
while the damage we are repairing -- linear attention replacing softmax, a 4x
compressed KV cache, no positional encoding -- is precisely long-range. The
teacher sees the same concatenated text, so the KL target is well defined; it
just does not exercise retrieval.

Filtering rather than switching corpora keeps the distribution identical, which
matters because the recovery objective is distillation from a teacher that was
not trained on books or papers. Storage stays small: only documents that pass
are written, so the disk cost is the output size, not the streamed size.

Yield is the cost. At ~0.8% of documents over 8192 tokens, reaching 30 M tokens
of long documents means streaming on the order of 10^8 tokens. That is network
time, not disk.
"""
import argparse
import os
import shutil
import sys
from mercurius.paths import STAGE_AB
OUT_DEFAULT = "data/fineweb_edu_long.txt"
MIN_FREE_GB = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tokens", type=int, default=4096,
                    help="keep documents at least this long. 4096 rather than "
                         "8192 because yield falls off a cliff: p90 is 2041.")
    ap.add_argument("--target-tokens", type=int, default=30_000_000)
    ap.add_argument("--max-docs-scanned", type=int, default=2_000_000,
                    help="stop regardless, so a bad yield cannot run forever")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--max-out-mb", type=float, default=400.0,
                    help="hard cap on what is written; the root filesystem is "
                         "the only filesystem on this board")
    a = ap.parse_args()

    free = shutil.disk_usage("/").free / 1e9
    if free < MIN_FREE_GB + a.max_out_mb / 1000:
        raise SystemExit(f"refusing to start: {free:.1f} GB free, need "
                         f"{MIN_FREE_GB + a.max_out_mb/1000:.1f} GB")
    print(f"disk {free:.1f} GB free; cap {a.max_out_mb:.0f} MB", flush=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(STAGE_AB))

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    kept = kept_tokens = scanned = 0
    bytes_written = 0
    cap = a.max_out_mb * 1e6
    # cheap prefilter: 3.5 chars/token is a safe underestimate, so anything
    # shorter than this cannot possibly reach min_tokens and never gets tokenized
    min_chars = int(a.min_tokens * 3.0)

    with open(a.out, "w") as fh:
        for rec in ds:
            scanned += 1
            txt = rec.get("text") or ""
            if len(txt) >= min_chars:
                n = len(tok(txt, add_special_tokens=False).input_ids)
                if n >= a.min_tokens:
                    fh.write(txt.rstrip() + "\n\n")
                    bytes_written += len(txt) + 2
                    kept += 1
                    kept_tokens += n
            if scanned % 20000 == 0:
                pct = 100 * kept / scanned
                print(f"  scanned {scanned:,}  kept {kept:,} ({pct:.2f}%)  "
                      f"{kept_tokens/1e6:.1f} M tok  {bytes_written/1e6:.0f} MB",
                      flush=True)
            if kept_tokens >= a.target_tokens:
                print("  reached target"); break
            if bytes_written >= cap:
                print("  reached size cap"); break
            if scanned >= a.max_docs_scanned:
                print("  reached scan limit"); break

    mb = os.path.getsize(a.out) / 1e6
    print(f"\nwrote {a.out}  {mb:.0f} MB  {kept:,} docs  "
          f"{kept_tokens/1e6:.1f} M tokens", flush=True)
    if kept:
        print(f"  mean {kept_tokens/kept:.0f} tokens/doc "
              f"(current corpus: 934 mean, 550 median)")
        print(f"  yield {100*kept/max(scanned,1):.2f}% of documents scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
