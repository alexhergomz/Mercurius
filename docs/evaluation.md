# Evaluation: what we measure, and why contamination decides the shape of it

The base Qwen3.5-4B was pretrained on an undisclosed corpus. So any score on a
benchmark that existed before its cutoff partly measures memorisation, and we
cannot tell how much. That is not a reason to skip those benchmarks -- they are
how anyone compares to anyone -- but it means the suite has to be built around
sources where a good score CANNOT come from having seen the answer.

The property worth paying for is a **released generator**: unlimited fresh
instances, so contamination stops being a question rather than being argued
about. Fixed test sets age; generators do not.

## 1. Ours, and already procedural

**Held-out AST task split** (`experiments/capability_eval.py`, 77 tasks over 10
repositories). Questions are synthesised by PARSING a repository -- "where is
`getWrittenXRefTable` called" has never existed in any corpus because we made it
from the syntax tree. The code is public and the base model may have seen it; the
question and the answer format cannot have been. 11,258 candidate symbols across
78 repositories means we can regenerate indefinitely.

Split is by REPOSITORY, not by task: a model trained on other symbols from the
same repo has already seen its layout and naming, so a task-level split leaks.

Scored by `code_graph.verify`, the same oracle that admits training data.

**Ceiling caveat:** both the 27B and 122B teachers score near 100% on `locate`
(12/12 on one repository), so the easy kinds cannot separate models. Report per
kind; `impact` (multi-file aggregation) is the one that discriminates.

## 2. Generators worth adopting

| benchmark | licence | what it generates |
|---|---|---|
| **RULER** (NVIDIA) | Apache-2.0 | needle-in-haystack, multi-hop variable tracking, aggregation, at ANY context length from a seed. Rule-graded, no judge. |
| **DyVal** (in microsoft/promptbench) | MIT | graph-informed procedural reasoning -- boolean/arithmetic expression trees, Dyck structures, controllable complexity. Built to defeat contamination. Repo archived but the code works. |

**We already had RULER and were using it wrongly.** It is a generator; the Paul
Graham essay haystack was our choice, not its requirement. Swapping the filler
for code fixes three things at once: the domain mismatch (we measure retrieval in
English essays for a model that retrieves symbols from repositories), the
copyright flag already in data/ATTRIBUTION.md ("the essays are the author's
copyright ... replace with public-domain text"), and contamination.

**DyVal is the STEM generator we could not find as a dataset.** After rejecting
every open maths corpus on provenance grounds (docs/data_policy.md S11), a
procedural generator is the clean answer for evaluation, and possibly for
training too.

## 3. Continuously refreshed (filter by date)

| benchmark | licence | refresh | grading |
|---|---|---|---|
| **LiveCodeBench** | MIT | rolling, problems carry release dates (`--start_date`) | execution; also has an execution-PREDICTION mode |
| **LiveBench** | Apache-2.0 / MIT | monthly | objectively verifiable, no LLM judge for maths/code |
| **SWE-bench-Live** | MIT | ~50 newly verified issues/month, 431 repos, 8 languages | pure test execution |
| **MathArena** | MIT | monthly competition maths | rule-graded on AIME/HMMT/Project Euler; the proof-based tracks need a judge, so avoid those |

**Correction to an earlier decision:** we rejected SWE-bench-Live and SWE-rebench
for TRAINING because their task statements are scraped GitHub issue prose. That
reasoning does not extend to EVALUATION -- MIT permits it, we redistribute
nothing, and the grading is execution. Rejecting them for training was right;
rejecting them for evaluation would be overcautious.

Before trusting any rolling benchmark as "unseen", check that its newest release
actually covers dates after the base model's cutoff.

## 4. Comparable but memorisable -- use, and caveat

| benchmark | licence | note |
|---|---|---|
| HumanEval | MIT | public since 2021; a good score is partly recall |
| MBPP | CC-BY-4.0 | same |
| GSM8K | MIT | same |
| EvalPlus (HumanEval+/MBPP+) | Apache-2.0 | 80x/35x more tests. A RIGOUR multiplier, NOT an anti-contamination measure -- the problem statements are the same old public prompts |
| CRUXEval | MIT | predict input/output rather than write code -- close to what our corpus trains -- but a fixed 2023 set, so three years to leak |
| BigCodeBench | Apache-2.0 | tool-call-heavy; static release with a build-time decontamination pass |

These are for comparability with published numbers, not for truth.

## 5. Rejected

* **GSM-Symbolic** (Apple) -- custom proprietary licence, not standard permissive,
  AND the generator is explicitly not released: only ~50 fixed instances per
  template, circulating since late 2024. Neither permissive nor fresh.
* **GSM-Plus** -- CC-BY-SA (share-alike), fixed set, and it perturbs KNOWN GSM8K
  items, so the base questions remain memorisable.
* **tau-bench** -- grading needs an LLM as both simulated user and fault
  classifier, originally GPT-4-class. Violates "no closed judge" unless we
  self-host an open one, and the authors themselves caveat fidelity.

## 6. Recommended stack

1. **Held-out AST tasks** (ours, procedural) -- the capability our corpus targets
2. **RULER with a code haystack** (generator) -- long-context retrieval, fixed domain bias
3. **DyVal** (generator) -- symbolic and arithmetic reasoning
4. **LiveCodeBench** filtered past cutoff -- code generation and execution prediction
5. **SWE-bench-Live** -- agentic coding, execution-graded, month-fresh
6. HumanEval / MBPP / GSM8K -- comparability only, always reported with the
   contamination caveat

## 7. Unverified, worth a look

MATH-Perturb (repo 404'd on the obvious URL; arXiv:2502.06453), SWE-rebench-V2's
refresh cadence, and the Berkeley Function-Calling Leaderboard's licence -- BFCL
is the best tool-use option found but its licence was not confirmed.

## Generation budget is part of the measurement (2026-09-24)

`--max-new 768` was passed for speed on the no-think full run. On MBPP that cost
up to 6.6 points: 17 of 100 failures ran to the ceiling mid-function. Measured,
not guessed -- median output is 661 chars on problems the model gets right and
1600 on ones it gets wrong, because an unsure model reasons in prose *in the
answer* even with `enable_thinking=False`. So the budget interacts with
difficulty: a cap that is generous for the easy half silently truncates the hard
half, which biases the score downward exactly where the score is contested.

Rules now:
  * Do not override `MAXNEW` (32,768) to make a run finish sooner. Use `--limit`
    with its seeded random sample instead -- shrink the *number* of problems,
    never the room each one gets.
  * A truncation count is reported per batch and any run with >2% truncation on
    a code task is not a clean number, whatever it says.
  * `enable_thinking=False` suppresses the `<think>` block, NOT the reasoning.
    The model still deliberates in prose. Budget for that.

## Three more scorer bugs, and why the validator missed them (2026-09-24)

Found while re-auditing after the budget mistake above. All three were
pre-existing except the first, and all three pushed scores DOWN.

1. **Generation budget.** `--max-new 768` truncated 8/164 HumanEval, 72/257
   MBPP, ~12/512 GSM8K -- every task above the 2% threshold. See above.

2. **`extract_code` returned the FIRST fence.** `m[0]` scores whatever the model
   tried first. 44 of 257 MBPP generations emitted more than one fence; one
   emitted 74. On real data first-vs-last was only +0.6 / -0.8 points, which
   looked like a judgement call -- but on a controlled draft-then-correct input
   first-fence scores **0/30**. The real-data delta was small only because
   truncation was masking it: the multi-fence generations are mostly the
   ramblers that got cut off anyway. Lesson: do not tune an extractor on
   contaminated data, and do not read a small delta on dirty data as "no bug".
   Fixed to take the last fence that DEFINES something (models often close with
   a usage example that defines nothing), never "try all fences and pass if any
   works" -- that is best-of-N over the model's own rejected drafts.

3. **The no-fence fallback dropped module-level state.** Anchoring at the first
   `def`/`import` cut `NO_OF_CHARS = 256` off MBPP/18, so four correct functions
   died on NameError. Bare assignments and decorators now anchor too. Affects
   the unfenced generations (6 HumanEval, 16 MBPP here).

**Why `validate_benchmarks.py` passed 14/14 while bugs 2 and 3 were live:** it
only ever built inputs with ONE fence and only ever used solutions with no
module-level state. A validator catches the shapes it thinks to build. Every
shape observed in real generations is now a case (multi-fence draft-then-final,
unclosed fence, both-fences-wrong), and it is 20/20.

Rescoring the saved generations with the fixed extractor, still truncated:
HumanEval 67.7% -> 68.3%, MBPP 61.1% -> 61.5%.

## Repairing a run: truncated only, never failed (2026-09-24)

`experiments/bench_repair.py`. Re-running everything that FAILED and keeping
what now passes is best-of-2 on the losing subset: it can only move the score up,
it does so for a model that got luckier rather than better, and the result is not
pass@1 on anything. The script only accepts `--only truncated`.

Truncation is a missing measurement, not a failed one -- the harness cut the
generation off, the model never said it was done. Retaking it is a repair,
provided: selection uses only budget exhaustion (independent of correctness),
the retake's verdict stands even when it turns a pass into a fail, and both
numbers plus the count of touched items are reported.

## Greedy is a diagnostic tool, never a measurement (2026-09-24)

The benchmarks sample: T=0.7 / top_p=0.8 / top_k=20, which is Qwen3's stated
non-thinking setting (thinking is 0.6 / 0.95 / 20). `--temperature 0` is now
REJECTED at the CLI -- the model cards forbid greedy decoding because it
degenerates, and at 4B it degenerates readily.

`temperature <= 0 -> argmax` survives inside `_sample` for one legitimate use:
comparing two DECODE PATHS (cached vs full recomputation) requires removing the
sampler as a variable. That use is sound.

**The inference it invites is not.** Having run the cache comparison greedily, I
read the CONTENT of its output -- the model rewriting its own planning comments
-- as evidence that repetition was "real model behaviour rather than a decoding
artifact". That is backwards: greedy repeating itself is the expected behaviour
at this scale and is evidence of nothing about the sampled run. The only valid
evidence about the sampled run is the sampled run, where 6 of 421 generations
(1.4%) had exactly-periodic tails, all of them inside truncated generations and
none among the 341 that terminated.

Rule: a greedy run answers "do these two code paths agree". It never answers
"does the model do this".

## Measuring repetition: three detectors, two wrong answers (2026-09-24)

A user challenge -- "text similarity itself doesn't necessarily mean a bad cycle,
you do this yourself when reasoning" -- was correct, and only reading the
generations settled it. The same 421 generations, three metrics:

| detector | "degenerate" | what it actually measures |
|---|---|---|
| exact token periodicity (last 200 tok) | 1.4% | literal cycles only; blind to a model restating itself in new words |
| duplicate-line fraction | 71% | ~collinear with "hit the cap"; scores a redraft the same as a copy |
| novel content in the final quarter | **12.4%** | whether the model is still making PROGRESS |

The first two were reported as fact before anyone looked at the text. They are
wrong in OPPOSITE directions, which is the tell: a metric that can only be
checked against another metric is not yet evidence.

Reading the generations found three distinct behaviours that the first two
detectors merge:
  * byte-identical function re-emitted repeatedly (mbpp 146, dup 0.90) -- the
    genuinely pathological case;
  * "Actually, let me reconsider the implementation... Let me fix this:" followed
    by a REVISED attempt (humaneval 44, dup 0.65) -- legitimate drafting, scored
    identically to the loop above;
  * one long unrepetitive deliberation that simply ran out of budget
    (mbpp 127, dup 0.000).

Novel-tail separates them, and validates cleanly: generations that TERMINATED
have a median novel-tail of 1.00 -- perfect separation from the stuck ones.

Breakdown of the 79 truncated: 35% stuck, 22% mostly repeating, 14% revising,
29% fully novel. So roughly 43% of truncated generations are doing real work and
should recover on a resumed repair; the stuck third will not.

Also note: duplicate-line fraction is ~0 at EVERY length below the cap
(0.000-0.004 from 0 to 760 tokens) and 0.407 at the cap. The failure is
bimodal -- the model either finishes cleanly or gets stuck -- not a gradual
degradation with length. Any "N times more repetitive" claim built on it is
really just the truncation rate restated.
