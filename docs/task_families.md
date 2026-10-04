# Task families: what we can verify, and what is worth teaching

The constraint is verification without execution (docs/decisions.md D1, D2): our
144 clones are 12+ languages, only 11 are installable Python with tests, and the
prebuilt SWE-smith images are x86 while this machine is aarch64. The question is
how much REAL software-engineering work can be made verifiable under that
constraint.

Measured 2026-09-23: the structural tasks we have are too easy. The 27B teacher
went 12/12 on Jaeger, and of the earlier pilot's failures, 4 of 8 were our own
broken oracle and 2 were formatting. A task solved 12/12 in 3-5 turns teaches a
competent student almost nothing.

## Two different goals, kept apart

An earlier draft of this document treated localization as a headline SWE task on
the strength of LocAgent's +12pp transfer number. That was a misreading of the
setting. LocAgent's input is a VAGUE HUMAN ISSUE, and localization is a
bottleneck there precisely because nothing in the input names the function. Our
input is a failing test, which names it. A developer holding a failing test does
not localize; they read the test and go and fix the thing.

Manufacturing difficulty by HIDING information a real developer would have
teaches a skill that does not exist in the workflow. So the tiers split by
purpose, and are never conflated:

* **Retention tiers (R)** -- parser-verified retrieval over a repository. These
  are tool-call and long-context training. Their job is to keep retrieval intact
  through latent-KV compression and NoPE, which is a real risk we introduced
  surgically. They are NOT software engineering, are kept narrow and controlled,
  and should be a minority of the mix.
* **Engineering tiers (S)** -- real defects, real fixes, with everything a human
  had at the time. These are the capability we actually want.

## Tier A (R) -- structural retrieval (built, weak)

`locate` / `callers` / `impact` from the parsed symbol graph. Ground truth is
exact and free. Keep as a small baseline tier and for measuring regressions, not
as the bulk of the corpus.

## Tier B (R) -- described-symbol retrieval (cheap upgrade, no new infrastructure)

Identical ground truth, but the question DESCRIBES the symbol instead of naming
it: "the function that applies the BGP daemon configuration" rather than
`applyGOBGPDConfig`. A named symbol is one grep; a described one demands semantic
search, reading candidates and disambiguating -- the capability `search_code`
exists for, and the one latent-KV compression and NoPE most endanger.

Descriptions come from the symbol's own signature and leading comment, which the
graph already stores in `self.sigs`. Guard: the description must not contain the
identifier or any ground-truth path.

## Tier C (R) -- real co-change impact (needs history)

Our synthetic `impact` answer is "definition + declarations + call sites", which
is what a parser can see. A real commit shows what a HUMAN had to touch: config,
fixtures, docs, generated files, a sibling implementation of the same interface.
Ground truth = the source files changed together in one commit that also touched
symbol S. Exactly verifiable as set equality, and it captures coupling static
analysis cannot.

## Tier S -- real fixes, by direct supervision (needs history; the main tier)

Give the agent the repository at the PARENT of a real bug-fix commit -- the buggy
state -- together with the failing test from that commit. That is exactly what
the developer had. Ask for the fix.

**Verification: none, because none is possible and none is needed.** A generated
patch cannot be checked without executing the test suite, which this machine
cannot do at scale (D1). But the commit IS the ground truth, so the task becomes
DIRECT SUPERVISION rather than rejection sampling: the target is the real diff,
written by the person who owned the code.

That inverts the economics of everything else here. Rollouts, rejection sampling
and bridging all cost teacher generation; this costs none. At ~15-20k focused fix
commits it is the largest source of real engineering signal available to us, and
the cheapest.

It fits the objective unchanged: the excess-nats formulation already carries a CE
data term, so a human-written target slots in there. Only the distillation terms
need teacher logits.

Cost: diffs must be materialised once. Treeless clones make traversal free but
each diff is a ~2.5 s network blob fetch (0.002 s once cached), so extraction is
a one-off parallel batch job of roughly an hour, cached to JSON.

### Tier S-agentic -- the same defect, worked through tools

The non-agentic form teaches what a fix looks like but not how to find one. The
agentic form keeps the tools: the agent explores the buggy repository, reads the
failing test, and proposes the patch. Exploration comes from the teacher and is
unverifiable; the FINAL patch is compared against the real fix at function
granularity.

That comparison is the localization signal -- but embedded inside a real task as
one component of it, rather than pulled out into an artificial task of its own.
It is a weak check (a different correct fix scores as wrong), so it is a data
FILTER, never a claim of correctness.

## Tier E (S) -- defect spotting

Parent state of the region a fix touched; what is wrong with it? Ground truth is
the lines the fix changed, scored by overlap. Closer to code review than
debugging, and the one place where withholding the test is legitimate, because a
reviewer genuinely does not have one.

## Tier F -- type-check verified edits (corrects D1, and the only route to
verifying generated patches here)

D1 dismissed execution too broadly. A repository's TEST SUITE needs its whole
dependency environment, which is what collapses the corpus to 11 repositories.
Type checkers do not: `mypy`, `tsc`, `go vet`, `cargo check` and `javac` are
ARM-native, fast, and mostly need only the source tree.

This is the only mechanism on this machine that can judge a patch the model
actually wrote. Necessary, not sufficient -- a patch can typecheck and be wrong --
but it turns Tier S-agentic from pure imitation into something with a real
gate. Scoped separately.

## What stays constant

Ground truth is never model-generated. Statements are derived from code, never
from commit messages (which are human prose, used only as a filter --
docs/data_policy.md). Every oracle is audited against an independent tool before
it is trusted at scale: the MAVSDK bug rejected correct answers AND accepted
incomplete ones, and nothing in the metrics showed it.

## Order

1. Tier S extraction (the main tier, cheapest signal, no generation cost).
2. Tier B (narrow retention tier, no new infrastructure).
3. Tier F scoping, which decides whether S-agentic can be gated.
4. C and E fall out of the S extraction.
