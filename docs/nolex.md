# NOLEX — NO LEXical overlap

A long-context recall benchmark where the question shares no content word with
the needle, so the model must infer a latent association instead of matching a
string.

## Motivation

Needle-in-a-haystack tests are easy because the needle and the question share
words. RULER's needle is "One of the special magic numbers for
`<adjective-noun>` is: `<number>`" and the question names that same
`<adjective-noun>`, so a model that can match a rare string can solve it without
understanding anything. NoLiMa (arXiv:2502.05167) showed that removing the
overlap collapses performance on models that look strong on NIAH.

NoLiMa's data and code are under the Adobe Research License, non-commercial.
Of the open long-context suites — RULER and BABILong and Zoology (Apache-2.0),
HELMET and InfiniteBench and LongBench (MIT) — none isolates this property.
BABILong is closest in spirit, but its questions still name the entities that
appear in its needles.

NOLEX is written from scratch so the result can be released under a permissive
licence. It borrows the idea, which is a benchmark design; the needle set,
templates, haystack, scoring and axes are new.

## What it measures

A fact chain: an **anchor** (a landmark), the **city** it stands in, the
**country** that city is in. The needle names the anchor. The question names the
anchor, the city, or the country, giving a graded overlap ladder where NoLiMa
has a binary one:

| level | question names | overlap with needle |
|---|---|---|
| L0 | the anchor, verbatim | literal string match |
| L1 | the city | none — one hop |
| L2 | the country | none — two hops |

L0 doubles as the feasibility control. The slope from L0 to L2 separates "cannot
find the needle" from "cannot make the association".

## Four differences from NoLiMa

**1. Per-fact feasibility calibration.** Every fact is first asked at minimal
context: the needle and the question, no haystack. A fact the model fails there
is testing its world knowledge, not its retrieval, and is dropped *for that
model*. Degradation is then measured only over facts the model demonstrably
knows. NoLiMa filters its needle set once, globally, with a 70B model, which
cannot know what a small model knows — and its smallest published model, Gemma 3
4B, already scores 16.4 at 4K and 2.3 at 16K. Calibrating per model is what lets
the benchmark run at any model size.

**2. Forced choice by likelihood, one forward per sample.** The answer is a name
from a pool verified to be single tokens under the tokenizer in use, so the whole
candidate set is read off the final position's logits. No generation, no
instruction-following, no judge. The candidate set is fixed at 10 however many
needles are planted, so the chance level does not move when an axis does.

**3. A measured floor.** Every condition also runs with the needles removed. A
model that never saw the needle does not answer at 1/10; it falls back to a prior
over names. NIAH and NoLiMa assume the floor, so a low score cannot be read as
above chance or not. NOLEX measures it.

**4. A state-load axis.** K independent facts are planted and one is queried, for
K = 1, 2, 4, 8, at fixed context length. For fixed-state operators, recall
degrades with the NUMBER of items retained, not only with distance — see Jelassi
et al. on copying (arXiv:2402.01032) and Arora et al. on the recall-state
tradeoff (arXiv:2312.04927, arXiv:2402.18668). Softmax attention pays no such
price. Neither NIAH nor NoLiMa sweeps this, and it is the axis most likely to
expose a linear-attention conversion.

## Reported quantities

Per (level, K, length): accuracy, P(gold) over the candidate set, and the logit
margin between the gold name and the best distractor. Margin is graded and has
no floor, so it orders models whose accuracy has bottomed out.

## Construction

Haystack sentences are shuffled, so no discourse structure survives that a model
could exploit to locate an out-of-place sentence. The queried needle sits at a
requested depth; the other K-1 are spread at random depths. Context length is
fitted by accumulating sentences to the token budget, so the nominal length is
the real one for the tokenizer in use.

## Limits

- The fact set is landmark/city/country only. One association type, many
  instances. Other types (tool/task, species/genus) fit the same schema.
- Facts are authored, not harvested, so the set is small by construction.
- Single-token candidate names are tokenizer-specific; the runner asserts the
  property rather than assuming it, and the pool must be re-checked for a
  different tokenizer.
