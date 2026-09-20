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


def make_doc(rng, sentences, k, approx_tokens, tok):
    """One document: k facts spread through real prose, restated at the end."""
    topics = rng.sample(TOPICS, k)
    codes = [_code(rng) for _ in topics]
    facts = [FACT.format(topic=t, code=c) for t, c in zip(topics, codes)]

    # real prose, enough to overshoot the window so the document earns a span
    body, n = [], 0
    while n < approx_tokens:
        s = sentences[rng.randrange(len(sentences))]
        body.append(s)
        n += len(s.split()) * 4 // 3          # rough words->tokens
    # spread the facts over the first 85%, so every one of them sits BEFORE the
    # summary by a large margin rather than adjacent to it
    for i, f in enumerate(facts):
        pos = int(len(body) * 0.05 + (i + 1) * len(body) * 0.80 / (k + 1))
        body.insert(min(pos, len(body)), f)

    order = list(range(k))
    rng.shuffle(order)                        # restate in a DIFFERENT order
    tail = [SUMMARY_HEAD] + [SUMMARY_LINE.format(topic=topics[i], code=codes[i])
                             for i in order]
    return " ".join([LEAD] + body) + "\n" + "\n".join(tail)


def build_corpus(src_path, out_path, tok, n_docs=120, k_range=(2, 8),
                 approx_tokens=8600, seed=0, src_chars=8_000_000):
    """Write n_docs synthetic documents, blank-line separated for the loader."""
    raw = open(src_path, encoding="utf-8", errors="replace").read(src_chars)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw)
                 if 40 < len(s.strip()) < 400]
    rng = random.Random(seed)
    docs, total = [], 0
    for i in range(n_docs):
        k = rng.randint(*k_range)
        d = make_doc(rng, sentences, k, approx_tokens, tok)
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
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(CKPT)
    n, d = build_corpus(a.src, a.out, tok, n_docs=a.n_docs)
    print(f"  wrote {d} documents, {n:,} tokens, mean {n//d:,} tok/doc -> {a.out}")
    print(f"  documents at least {a.seq} tokens get a sampling span")
