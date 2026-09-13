"""Fetch a small real-text corpus for perplexity, and check max feasible context."""
import sys, os, time, torch
from mercurius.paths import FINEWEB, STAGE_AB, WIKITEXT
DATA = str(WIKITEXT)


def fetch():
    if os.path.exists(DATA) and os.path.getsize(DATA) > 200_000:
        print(f"already have {DATA} ({os.path.getsize(DATA)/1e6:.1f} MB)")
        return
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    open(DATA, "w").write(text)
    print(f"wrote {DATA} ({len(text)/1e6:.1f} MB chars)")


def feasibility():
    from transformers import AutoTokenizer
    from mercurius.models.kda import load_kda_model
    CKPT = str(STAGE_AB)
    tok = AutoTokenizer.from_pretrained(CKPT)
    text = open(DATA).read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    print(f"corpus tokens: {len(ids):,}")

    model = load_kda_model(CKPT, dtype=torch.bfloat16)
    print("probing max context (bf16) ...", flush=True)
    for n in (512, 2048, 8192, 16384, 32768):
        if len(ids) < n:
            print(f"  {n:>6}: corpus too short"); continue
        try:
            x = ids[:n].unsqueeze(0).cuda()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=x)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  {n:>6}: {dt:6.2f}s  peak {peak:5.2f} GiB  "
                  f"loss-ready {tuple(out.logits.shape)}", flush=True)
            del out, x
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except RuntimeError as e:
            print(f"  {n:>6}: FAILED — {str(e)[:120]}")
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            break




def fetch_fineweb(target_mb=120):
    """Training corpus: FineWeb-Edu, kept SEPARATE from the wikitext eval corpus
    so perplexity measures recovery rather than memorization."""
    out = str(FINEWEB)
    if os.path.exists(out) and os.path.getsize(out) > target_mb * 900_000:
        print(f"already have {out} ({os.path.getsize(out)/1e6:.1f} MB)")
        return out
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf, n = [], 0
    for rec in ds:
        t = rec.get("text", "")
        if not t:
            continue
        buf.append(t); n += len(t)
        if n > target_mb * 1_000_000:
            break
    open(out, "w").write("\n\n".join(buf))
    print(f"wrote {out} ({n/1e6:.1f} MB, {len(buf):,} docs)")
    return out


if __name__ == "__main__":
    if "--fineweb" in sys.argv:
        fetch_fineweb()
    else:
        fetch()
        feasibility()
