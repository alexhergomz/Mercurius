"""Synthetic documents that require holding several facts at once.

WHY THIS IS CHEAP. Training here is distillation against a live teacher, so no
labels are needed -- only INPUTS that exercise the capability. The teacher is the
unmodified model, which scores 98.8% multi-needle exact match on RULER while the
converted student scores 79-85%. The teacher already knows how to do this. Feeding
text whose correct continuation requires retaining several earlier facts puts KL
pressure exactly where the student is wrong, and the supervision comes free.

Nothing in the training mix has ever required this, which is why it is the first
untested lever worth pulling: the failure is multi-item retention and the corpus
has never asked for it.

HELD OUT FROM THE BENCHMARKS BY CONSTRUCTION. Training on the eval format would
make RULER and NOLEX meaningless, so every surface feature differs:

    RULER    Paul Graham essays, "One of the special magic numbers for
             <adjective-noun> is: <7-digit number>", question + answer prefix
    NOLEX    shuffled fineweb sentences, "<Name> has lived right beside <landmark>",
             question naming the city or country
    here     contiguous fineweb documents, "Reference code for <topic>: <hex>",
             no question at all -- a closing summary block that RESTATES the codes

The restatement is the mechanism. There is no query and no answer prefix; the
document simply ends with a summary listing the codes in a different order from
the one they were introduced in. Predicting those tokens requires having carried
k facts across thousands of tokens, and gets that from the teacher's distribution
rather than from a hand-written label.

Each document is built to exceed the training sequence length so that
tokenize_by_document gives it a span and a window drawn inside it stays inside it.
"""
import random
import re
from mercurius.paths import DATA_DIR

TOPICS = [
    "the north annex", "the coastal survey", "the winter ledger",
    "the third revision", "the harbour census", "the upper reading room",
    "the quarry report", "the eastern terminal", "the autumn inventory",
    "the tidal register", "the mill accounts", "the border concordance",
    "the lantern schedule", "the granary audit", "the southern circuit",
    "the copper tables", "the orchard returns", "the ferry manifest",
]
FACT = "Reference code for {topic}: {code}."
LEAD = ("The following notes include reference codes. They are restated in the "
        "summary at the end.")
SUMMARY_HEAD = "Summary of reference codes."
SUMMARY_LINE = "{topic} -> {code}"


def _code(rng):
    return "".join(rng.choice("0123456789ABCDEF") for _ in range(6))


def make_doc(rng, sentences, k, approx_tokens, tok, period_tokens=2000,
             lags=(0, 1, 2, 3)):
    """Periodic facts, restated at SEVERAL distances.

    The first working corpus cycled facts and a restatement inside one ~2,600
    token block, so every retention distance it taught was ~1,800-2,300. The model
    learned exactly that: measured at n=40 on multivalue, 99.4% at 4k, 87.5% at
    8k, 64.4% at 16k -- near-saturated where the corpus taught and weakest where
    it did not (findings 0.5).

    Here group g's facts are planted at the start of period g, and the END of each
    period restates the groups `lags` periods back. With period 2000 and lags
    (0,1,2,3) the distances taught are roughly 2k, 4k, 6k and 8k, spanning the
    whole training window instead of clustering at its short end.

    Restatements still use a shuffled order, so predicting them requires having
    retained the facts rather than copying a nearby span.
    """
    n_periods = max(len(lags) + 2, approx_tokens // period_tokens)
    groups = []
    parts = [LEAD]
    for g in range(n_periods):
        topics = rng.sample(TOPICS, k)
        codes = [_code(rng) for _ in topics]
        groups.append((topics, codes))
        body, n = [], 0
        while n < period_tokens:
            s = sentences[rng.randrange(len(sentences))]
            body.append(s)
            n += len(s.split()) * 4 // 3
        # facts at the START of the period, so the distance to a restatement at
        # the end of period g+lag is about (lag+1) * period_tokens
        for i, (tp, cd) in enumerate(zip(topics, codes)):
            pos = int(len(body) * 0.02 + i * len(body) * 0.06 / max(k, 1))
            body.insert(min(pos, len(body)), FACT.format(type_needle_v="numbers",
                                                         topic=tp, code=cd)
                        if False else FACT.format(topic=tp, code=cd))
        parts.append(" ".join(body))
        for lag in lags:
            src = g - lag
            if src < 0:
                continue
            tps, cds = groups[src]
            order = list(range(len(tps)))
            rng.shuffle(order)
            parts.append(SUMMARY_HEAD + " "
                         + "; ".join(SUMMARY_LINE.format(topic=tps[i],
                                                         code=cds[i])
                                     for i in order))
    return " ".join(parts)


def build_corpus(src_path, out_path, tok, n_docs=120, k_range=(2, 8),
                 approx_tokens=16000, seed=0, src_chars=8_000_000,
                 period_tokens=2000, lags=(0, 1, 2, 3)):
    """Write n_docs synthetic documents, blank-line separated for the loader."""
    raw = open(src_path, encoding="utf-8", errors="replace").read(src_chars)
    # Collapse ALL whitespace inside a sentence. "\n\n" is the document
    # separator tokenize_by_document splits on, so a source sentence containing a
    # blank line silently splits the document it lands in: the first corpus wrote
    # 300 documents and the loader saw 476 fragments, median 5,988 tokens, 63%
    # under the 8192 span threshold. Fragments also break the task itself --
    # facts end up separated from the summary that restates them.
    sentences = [re.sub(r"\s+", " ", s).strip()
                 for s in re.split(r'(?<=[.!?])\s+', raw)]
    sentences = [s for s in sentences if 40 < len(s) < 400]
    rng = random.Random(seed)
    docs, total = [], 0
    for i in range(n_docs):
        k = rng.randint(*k_range)
        d = make_doc(rng, sentences, k, approx_tokens, tok,
                     period_tokens=period_tokens, lags=lags)
        docs.append(d)
        total += len(tok(d, add_special_tokens=False).input_ids)
    with open(out_path, "w") as fh:
        fh.write("\n\n".join(docs))
    return total, len(docs)


if __name__ == "__main__":
    import argparse, sys
    from transformers import AutoTokenizer
    from mercurius.recovery.train import CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA_DIR / 'fineweb_edu_long.txt'))
    ap.add_argument("--out", default=str(DATA_DIR / 'synth_recall.txt'))
    ap.add_argument("--n-docs", type=int, default=120)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--approx-tokens", type=int, default=16000,
                    help="target length of each synthetic document")
    ap.add_argument("--period-tokens", type=int, default=2000)
    ap.add_argument("--lags", default="0,1,2,3",
                    help="restate the groups this many periods back; distance "
                         "taught is about (lag+1) * period-tokens, so "
                         "0,1,3,7,15 at period 2000 teaches 2k..32k")
    ap.add_argument("--src-chars", type=int, default=8_000_000)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(CKPT)
    lags = tuple(int(x) for x in a.lags.split(","))
    n, d = build_corpus(a.src, a.out, tok, n_docs=a.n_docs,
                        approx_tokens=a.approx_tokens,
                        period_tokens=a.period_tokens, lags=lags,
                        src_chars=a.src_chars)
    print(f"  retention distances taught: "
          f"{', '.join(f'{(l+1)*a.period_tokens//1000}k' for l in lags)}")
    print(f"  wrote {d} documents, {n:,} tokens, mean {n//d:,} tok/doc -> {a.out}")
    print(f"  documents at least {a.seq} tokens get a sampling span")
