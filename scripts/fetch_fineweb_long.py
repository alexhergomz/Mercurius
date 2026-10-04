"""Fetch MORE long FineWeb-Edu documents for the long run (#61: text 70% of 100M tokens).

Keeps documents of at least --min-tokens (8192: one full training window, so every
window lies inside ONE document -- the reason fineweb_edu_long exists), in the SAME
file format as data/fineweb_edu_long.txt: documents separated by a blank line, internal
blank lines collapsed to single newlines (tokenize_by_document splits on blank lines).
De-duplicated against the existing file by a hash of each document's first 400
normalised characters. Unlike the original file, METADATA IS KEPT (id, url, dump, score,
int_score, n_tokens) in a sidecar JSONL -- data_policy 12.1 notes its value.

Licence: FineWeb-Edu, ODC-By 1.0 (admissible, data_policy 12.1).

    .venv/bin/python scripts/fetch_fineweb_long.py --target-tokens 40000000 \
        --out data/fineweb_edu_long_more.txt
"""
import argparse
import hashlib
import json
import re

BLANK = re.compile(r"\n\s*\n+")


def norm(t):
    return BLANK.sub("\n", t.strip())


def key(t):
    return hashlib.md5(re.sub(r"\s+", " ", t[:400]).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default="data/fineweb_edu_long.txt")
    ap.add_argument("--config", default="sample-100BT")
    ap.add_argument("--min-tokens", type=int, default=8192)
    ap.add_argument("--target-tokens", type=int, default=40_000_000)
    ap.add_argument("--out", default="data/fineweb_edu_long_more.txt")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from mercurius.paths import STAGE_AB
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    seen = {key(d) for d in open(a.existing, encoding="utf-8", errors="replace").read()
            .split("\n\n") if d.strip()}
    print(f"existing documents: {len(seen):,}", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=a.config, split="train",
                      streaming=True)
    total = kept = scanned = dup = 0
    min_chars = int(a.min_tokens * 3.2)        # cheap prefilter before tokenizing
    with open(a.out, "w", encoding="utf-8") as fo, \
            open(a.out.rsplit(".", 1)[0] + ".meta.jsonl", "w", encoding="utf-8") as fm:
        for r in ds:
            scanned += 1
            t = r["text"]
            if len(t) < min_chars:
                continue
            d = norm(t)
            k = key(d)
            if k in seen:
                dup += 1
                continue
            n = len(tok(d, add_special_tokens=False).input_ids)
            if n < a.min_tokens:
                continue
            seen.add(k)
            fo.write(("\n\n" if kept else "") + d)
            fm.write(json.dumps({"id": r.get("id"), "url": r.get("url"), "dump": r.get("dump"),
                                 "score": r.get("score"), "int_score": r.get("int_score"),
                                 "n_tokens": n}) + "\n")
            kept += 1
            total += n
            if kept % 200 == 0:
                print(f"  scanned {scanned:,}  kept {kept:,} docs  {total/1e6:.1f}M tokens"
                      f"  (dupes {dup})", flush=True)
            if total >= a.target_tokens:
                break
    print(f"done: {kept:,} documents, {total:,} tokens -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
