# Decisions log

Choices that were expensive to reach and are cheap to forget. Each records what
was decided, why, and what would have to change for the answer to flip.

## D1. Execution-verified SWE tasks (SWE-smith / R2E-Gym images) -- DEFERRED

SWE-smith publishes 59,136 task instances over 128 Python repositories, MIT, with
synthesised problem statements and a prebuilt Docker image per instance carrying
FAIL_TO_PASS / PASS_TO_PASS tests. Its *trajectories* are barred (Claude-generated,
see docs/data_policy.md S10) but its *tasks* are clean, and generating our own
trajectories against them would give real execution-verified reward.

Blocked on: **every image is `swesmith.x86_64.*` and this machine is aarch64**
(GB10 / DGX Spark). Running them needs qemu binfmt emulation, which is 5-20x slower
per test run and would have to carry thousands of rejection-sampling rollouts. Not
worth it.

Flips if: we get x86 compute, OR we rebuild environments natively on ARM for a
Python subset (SWE-smith gives repo+commit+install recipe, so this is mechanical
for maybe 20-40 repos), OR qemu turns out to be cheaper than assumed for
short-running test suites. Worth revisiting -- execution reward is strictly
stronger than static reward, we just cannot pay for it here today.

## D2. Verification by parsing, not by running -- ADOPTED

Our 144 clones are 12+ languages; only 11 are installable Python with tests. Any
execution-based scheme collapses the corpus to those 11 and reintroduces the
monoculture we removed in D4. tree-sitter parses all of them with nothing
installed: 144 repositories in 50 s, 78 of them yielding tasks, 11,258 candidate
symbols.

Supporting evidence (literature review, 2026-09-23): **LocAgent** (arXiv:2503.09089,
ACL 2025) builds exactly this -- repo-as-graph, localization as the task -- reaches
92.7% file-level accuracy fine-tuned on Qwen-2.5-Coder-32B, and critically
**improves downstream GitHub issue resolution by +12pp pass@10**. That is the only
hard transfer number found from a non-execution signal to real SWE performance, and
it is the shape we chose independently.

## D3. Unverified episodes quarantined, not deleted -- ADOPTED

3.53 M tokens / 201 episodes from the first build kept only structural filters
(answered, used tools, non-empty, mentions the expected path). No correctness
signal, so it distils the teacher's mistakes alongside its competence. Moved to
`data/episodes/unverified/` rather than deleted: V-STaR (arXiv:2402.06457) and
Re-ReST (arXiv:2406.01495) both show rejected/incorrect trajectories retain value
for verifier training and reflection-augmentation, and SRFT (arXiv:2605.10674)
argues whole-trajectory filtering discards usable steps inside failed rollouts.

## D4. Per-repository cap on third-party trajectories -- ADOPTED

Nemotron-SWE-v1 was 72% pandas across 5 repositories. Capped at 1 M tokens/repo:
5.66 M -> 3.34 M, 183 -> 68 trajectories. SWE-smith reports repository diversity
improving downstream performance **logarithmically at fixed dataset size**, which
supports paying tokens for breadth.

## D5. MoE teacher for generation -- ADOPTED (122B-A10B)

Decode on GB10 is memory-bandwidth-bound (~273 GB/s unified). Bytes read per token:
27B dense Q4 ~14.8 GB (~18 tok/s) vs 122B-A10B Q4 ~5.6 GB (~45 tok/s). The bigger
model is the faster one, and a better teacher. Same tokenizer and vocab (248320)
across the whole Qwen3.5 family, so no vocabulary surgery for logit distillation.

For the *training-time* teacher (in-process logits) the choice is 35B-A3B instead:
3B active, ~9x fewer active FLOPs than 27B dense. Blocked on loader work -- Qwen3.5
MoE stores experts as fused 3-D tensors `(256, hidden, 512)` which `bnb.Linear4bit`
cannot quantize as-is.

## D6. Task difficulty: a MIX, not a floor -- ADOPTED

"Where is symbol X defined" is often solvable with a single grep. That is NOT a
fairness problem and must not be filtered out: the agent uses only tools it would
really have, and finding an answer in one call instead of ten is the behaviour we
want. Rejecting efficient solutions would train performative thoroughness, which
is a worse agent.

The real risk is a corpus made entirely of one-hop lookups, which never exercises
the multi-file aggregation that `callers` and `impact` demand. So the lever is the
task-kind DISTRIBUTION and a measured spread of difficulty, not a minimum turn
count. Record turns-to-answer per task and keep the mix deliberately weighted
toward multi-hop rather than discarding the easy ones.

Distinguish this from real contamination, which is a live failure mode elsewhere:
63% of one frontier model's successful SWE-bench Pro resolutions retrieved the fix
from git history or mirrored pages (debugml.github.io/cheating-agents), and 176
SWE-bench Lite patches scored as passing were wrong (UTBoost, arXiv:2506.09289).
Those are cases where the information would not exist at the time a real agent
works the task. Ours is parsed from the checkout the agent is already looking at.

Evidence on what correlates with agent success (arXiv:2604.02547): exploration
before *editing* (rho +0.68) and validation effort (rho +0.50) help; jumping
straight to a patch hurts (rho -0.78). That is about premature action, not about
answering a lookup in few calls.

## D7. Execution verification IS possible here -- D1 was wrong

D1 said execution-verified tasks were out of reach. Two separate errors:

1. **The x86 blocker was only about PREBUILT IMAGES.** Running tests is fine:
   pre-commit-hooks installs and runs 417 tests in 1.8 s on aarch64, and our
   fail-to-pass gate verified 3 of 8 real fix commits at ~3 s each. SWE-bench's
   own README documents ARM support the same way -- rebuild locally from the
   per-instance Dockerfile instead of pulling an image.
2. **"Only 11 repos" was not the constraint it sounded like.** SWE-Gym's entire
   dataset is 11 repositories, and that is where its +12-14pp came from. A small
   execution-verified core beside a broad unverified corpus is the field's shape.

And the tier is no longer capped at 11. SWE-smith publishes ~847 REPO PROFILES
(MIT) across 11 languages, each with plain `install_cmds` and `test_cmd` shell
strings plus per-repo log-parser overrides; the Docker coupling is confined to
`build_image()`, so the commands run on ARM bare metal. Of their 849 repositories,
**617 pass our permissive-only filter** and are not SWE-bench Verified:

  cpp 118, typescript 108, python 98, java 86, golang 74, javascript 69,
  rust 43, ruby 16, php 3, c 1, csharp 1
  (MIT 338, Apache-2.0 210, BSD-3 40, BSD-2 11, Unlicense 7, others 8)

Provenance stays clean because we take only their MIT-licensed SHELL COMMANDS.
Their problem statements are written by claude-3-7-sonnet and their repo pool
includes GPL projects -- both excluded here by our own filter. Fix commits are
mined from each repository's real history and verified by our own gate.

Still owed before use: Stack opt-out check on the 617, and history fetches
(~2 GB extrapolating from 282 MB for 78).

## D8. Parse structured test output, not console text -- ADOPTED

We were regexing `pytest -rA` console output. SWE-bench does the same and has a
`[100%]` progress-bar phantom test permanently baked into two of its gold
PASS_TO_PASS lists as a result. Use `pytest-json-report` for Python, `go test
-json` natively, `cargo-nextest` -> JUnit XML for Rust, and `junitparser` as the
cross-language normaliser; keep regex only as the fallback for runners that
cannot emit structured output.

## D9. C-extension repos need a rebuild step -- OPEN

msgspec scored 0/8, every one `fix_did_not_pass`, where pure-Python repos work.
It is a C-extension package and the editable install almost certainly does not
recompile when a different commit is checked out, so the tests run against stale
binaries. Unfixed, this silently yields nothing for every compiled repository --
the same shape of failure as the MAVSDK oracle bug: a wrong answer that looks
like a clean negative.

## D10. RL sampling heuristics do not transfer unexamined -- CORRECTED

Dynamic sampling was implemented with DAPO's justification: all-pass and
all-fail groups have zero advantage, therefore zero gradient, therefore drop
them (arXiv:2503.14476, arXiv:2504.11343). **That reasoning does not apply to
us.** The recovery objective is distillation -- `_chunk_div_terms` builds full
vocabulary teacher and student distributions at every position -- so every
sequence yields a dense gradient whatever its outcome. There is no advantage
term and no degenerate group.

The behaviour was kept; the justification was rewritten, because a wrong
rationale in a comment becomes a wrong decision later:

  * **all-fail -> stop early**: training on trajectories the teacher got wrong
    distils its mistakes. Data quality, not gradient.
  * **all-pass -> stop early**: budget only. We need `keep_per_task` sequences
    and the rest are near-duplicates. The task is good data.
  * **"mixed tasks are the informative ones"**: FALSE here. Every correct
    trajectory carries the same dense signal.

Measured at fixed generation budget (the binding constraint, since ~11k
candidate tasks are queued): 11,253 tasks attempted vs 9,177, 7,149 solved vs
6,774, same rollouts.

Two distillation-specific consequences to carry forward:

 1. Filtering teacher rollouts by outcome means we distil a FILTERED teacher,
    which is better than the teacher itself -- desirable, but "excess nats" is
    then measured on sequences selected by outcome rather than by teacher
    likelihood.
 2. Human-written commit diffs are NOT distillation targets. The teacher may
    assign them low probability, so they belong in the CE data term and should
    be weighted and monitored separately from teacher-generated sequences.

## D11. Rationalize the all-fail set from the gold diff -- BUILT, NOT RUN

The all-fail tasks are the hardest material we have and, in the execution tier,
the only ones where the correct answer is already known: the human's commit. No
rollout reached it, so there is nothing to distil. Rationalization recovers
them -- give the generator the gold diff as a HINT, have it do the investigation
that leads there, keep the investigation.

This is STaR's rationalization (Zelikman et al., arXiv:2203.14465), whose entire
technique is the clause people skip: train on the rationale WITHOUT the hint.

Guards (mercurius/data/rationalize.py), all tested:
  * the hint lives only in the generator's prompt; the stored sequence is
    rewritten to carry the ordinary task statement;
  * leakage checked three ways -- verbatim diff lines, the phrases a model uses
    when it knows the answer, and the same check over `reasoning_content`, since
    a leak inside hidden thinking is the one that would otherwise survive;
  * the generator authors assistant turns only; every tool result is replayed
    against the real environment;
  * the trajectory must still pass the ordinary oracle.

Recorded weakness, which no pass rate will show: a rationalized trajectory can
reach the right answer by reasoning that does not support it, because the
generator is working backwards from something it was told. Stored with
`rationalized: True` so any run using them can be ablated against one that does
not.

## D12. Top-k teacher caching cannot serve reverse KL -- MEASURED, REJECTED

Caching the teacher looked like the big efficiency win: its forward pass is ~69%
of a training step (54 GFLOP/token for the 27B against 24 for the student's
forward and backward), its output never changes, and no gradient flows through
it. Measured before generating, which is what saved the hours.

Verified against exact full-vocabulary terms on real model log-probs
(experiments/test_logit_cache.py, logit_cache.py):

  exact     revKL 0.0554   fwdKL 0.0276   excessCE 0.0447 nats
  k=64 u8   err   0.0550         0.0067          0.00013
  k=256 u8  err   0.0293         0.0029          0.00013

The reverse-KL error is the SIZE OF THE QUANTITY. Worse, with a student
identical to the teacher -- true reverse KL exactly zero -- the cache reports
0.0144 nats of divergence that does not exist, and at near-convergence
(T=1.05, true revKL 0.0009) the artefact is 20x the signal. It does not shrink
as training converges, so it is a floor precisely where resolution is needed.

The cause is structural. KL(s||t) = sum_i s_i (log s_i - log t_i) weights by the
STUDENT's mass, and even a perfectly matched student keeps ~0.3% of its mass
outside the teacher's top-64, where the reconstruction substitutes
log(tail_mass/(V-k)) ~ -18 for a true value nearer -8. **No cache of fixed
support can be unbiased for reverse KL**, because the student's support is
unknown at cache time and moves during training. Larger k shrinks the constant
(64 -> 256 halved it) but cannot remove it.

What survives:
  * **excess CE caches exactly** (1e-4 nats): one token's log-prob, nearly
    always inside top-k. The commit-diff tier needs no divergence term, so it
    becomes free on both axes -- no generation AND no teacher forward.
  * **forward KL caches acceptably** (24% error at k=64): teacher-weighted, so
    the teacher's own top-k captures what it reads.
  * 8-bit quantization is a non-issue -- u8 and f16 differ in the 5th decimal
    on every term. Truncation was always the risk, never precision.

Consequence: the efficiency answer for the divergence term is the MoE teacher
(35B-A3B, 3B active, ~9x fewer FLOPs than 27B dense) -- exact rather than
approximate. The fused-3D-expert loader is now the highest-value performance
work (see D5).

## D13. Cache the teacher's HIDDEN STATE, int8 per channel -- MEASURED, ADOPTED

D12 rejected caching after top-k logits failed. The object was wrong, not the
idea. Logits are `h @ W_t^T`, so their rank is bounded by the hidden width, not
by the 248k vocabulary, and `_chunk_div_terms` already consumes `h_t_c` rather
than logits. Caching the hidden state is therefore a drop-in, and it is EXACT
before quantization.

Measured against exact full-vocabulary reverse KL:

    scheme                  B/tok (4B)   revKL err
    bf16 hidden                   5120       0.00%
    int8 per-channel              2560       0.05%
    PCA r=512 + int8               512       0.51%   <- per-batch basis, MISLEADING
    top-k k=1024 (D12)            5120      15-20%

PCA was then tested honestly: basis fitted on python+go, applied to held-out
domains.

    domain      r=256       r=512      r=1024    int8
    python     39.42%       0.21%       0.20%   0.277%
    go         19.58%       0.01%       0.19%   0.032%
    prose   41508.58%   25583.33%   16058.34%   0.496%   (held out)
    spanish 50127.87%   36010.11%   27750.24%   0.298%   (held out)
    math    51051.25%   26137.56%   14139.40%   0.439%   (held out)

**PCA does not generalise across domains at all.** Hidden states for prose,
Spanish and mathematics lie outside the subspace code occupies, and projecting
them onto it destroys them. The earlier 0.51% was a per-batch SVD measuring its
own fit. Fitted on our code-heavy corpus it would have validated beautifully and
silently corrupted every non-code sequence.

ADOPTED: int8 per channel on the teacher's final hidden state. 0.03-0.5% across
every domain, no basis, nothing to generalise wrong. For the 27B (H=5120) that
is 5 KB/token, 51 GB for 10M tokens, and it skips the entire teacher body --
only the lm_head matmul remains (2.5 of 54 GFLOP/token), ~2.9x faster steps.

PCA was then retested with a MIXED basis over 7 domains (FineWeb prose, python,
go, java, rust, spanish, math), evaluated on held-out halves -- the same fix
that made CARE whitening work for MLA. It still fails:

    domain            r=256      r=512     r=1024      int8
    prose_fineweb  6077.07%   3888.97%   1652.55%    0.192%
    python         7887.92%   4771.14%   1928.81%    0.441%
    go             6464.80%   3233.17%   1241.31%    0.216%
    java           6602.14%   3070.55%   1067.44%    0.094%
    rust           7711.88%   4007.57%   1260.05%    0.086%
    spanish         639.02%    264.34%     45.29%    0.110%
    math           1403.38%    413.69%    107.01%    0.028%
    variance kept: r=256 0.7653  r=512 0.8737  r=1024 0.9584

So the corpus was NOT the whole story. r=1024 of 2560 dimensions keeps only
95.84% of the variance -- the spectrum is nearly flat -- and the missing 4% is
catastrophic because lm_head amplifies small directions in h into large logit
differences that reverse KL reads. **The final hidden state has no exploitable
low-rank structure, for any calibration set.**

Why this differs from the MLA SVD, where the same fix DID work: there we
compress per-head K/V activations, which genuinely are low-rank -- that is what
makes CARE whitening and TransMLA possible. The final pre-lm_head hidden state
is the aggregated output of the whole network and is close to full rank by
construction. Same technique, different object, opposite answer.

ORDERING: the cache is tied to a specific teacher AND its lm_head. Settle the
teacher first (35B-A3B MoE, D5) or the cache is thrown away with it.

## D14. Evaluation must be built on generators, not fixed sets -- ADOPTED

The base model's pretraining corpus is undisclosed, so every score on a
pre-cutoff benchmark partly measures memorisation and we cannot say how much.
The suite is therefore built around sources where a good score cannot come from
having seen the answer. Full detail in docs/evaluation.md; the decisions:

* **Our held-out AST split is already the right shape** -- questions synthesised
  by parsing a repository never existed in any corpus. Split by REPOSITORY, not
  by task, because a model trained on other symbols from the same repo has seen
  its layout and naming.
* **RULER is a generator and we were using it wrongly.** The Paul Graham essay
  haystack was our choice, not its requirement. Replacing the filler with code
  fixes the domain mismatch (we measure retrieval in English essays for a model
  that retrieves symbols from repositories), the copyright flag already recorded
  in data/ATTRIBUTION.md, and contamination -- three problems, one change.
* **DyVal (MIT) is the STEM generator** we could not find as a dataset after
  rejecting every open maths corpus on provenance grounds (data_policy S11).
* **Adopt rolling benchmarks with date filters**: LiveCodeBench, LiveBench,
  SWE-bench-Live, MathArena (auto-graded subsets only).
* **Reversal:** SWE-bench-Live and SWE-rebench were rejected for TRAINING because
  their statements are scraped issue prose. That does not extend to EVALUATION --
  MIT permits it, nothing is redistributed, grading is execution. Rejecting them
  for training was right; rejecting them for evaluation was overcautious.
* **EvalPlus is a rigour multiplier, not an anti-contamination measure**: 80x more
  tests, but the same public 2021 problem statements.
* Rejected: GSM-Symbolic (proprietary licence AND generator unreleased),
  GSM-Plus (share-alike, perturbs known items), tau-bench (needs a closed judge).

## D15. Mask tool output from BOTH loss terms -- CORRECTED, ADOPTED

84.7% of our episode tokens are tool output; 8.4% are assistant turns. The loss
was applied to every token, so roughly twelve times more gradient taught the model
to reproduce file contents than to decide what to do. On the diff tier, 91% of the
gradient taught it to reproduce the buggy source file it had just been shown.

**I argued the DIVERGENCE term was fine unmasked** and only CE needed fixing, on
the grounds that reverse KL against the teacher on tool-output text is ordinary
distillation. That was wrong. Tool outputs are prefill context: the environment
inserts them and the model never generates them, so supervising their prediction
supervises a behaviour that does not occur. All three papers doing agentic
distillation with KL exclude observations from the divergence term
(arXiv:2505.13820, 2605.07725, 2505.17612).

The one ablation (2505.13820, ALFWorld, 340M): correct span masking 56.3% >
token-level 52.1% > flat 48.2% > RANDOM span masking 45.9%. Random masking is
worse than none, so "mask correctly" is the requirement, not "mask".

Implemented in `episode_ds.assistant_mask` by character offset in the rendered
text mapped through the tokenizer's offset mapping -- incremental prefix
rendering does not work, because the chat template refuses a prefix with no user
turn and inserts its own control tokens.

Full detail and the rest of the agentic-training gaps in docs/agentic_training.md.

## 16. The degeneration is exposure bias: we trained open-loop (2026-09-24)

**This confirms an existing diagnosis on a new axis; it is not a new finding.**
Roadmap fact 4 already stated it from RULER: "Teacher-forced likelihood and free
generation disagree ... The model assigns high probability to each gold answer
and then loses the thread emitting four of them in sequence. That is exposure
bias." What is new here is only that the same signature appears on code and maths
benchmarks, and that the BASE model shows none of it.

**Symptom.** The surgered+distilled 4B fails to terminate on ~12% of
generations, restating itself until the budget runs out. Truncation on MBPP is
28.0% against the unmodified NF4 base's 4.7% -- 6x, on identical prompts -- and
HumanEval is 67.7% against the base's 79.9%.

**Cause.** `logs/recovery-masked150.json`: `on_policy = 0.0`. Every one of the
150 steps was teacher-forced on human-written text. The model was never once
asked to continue from its OWN prefix, so at inference -- closed-loop on its own
output -- it drifts off the training manifold and has no learned behaviour for
recovering, or for emitting EOS, from there.

`--on-policy` defaults to 0.0 and was not passed IN THIS RUN. It has been used
before: roadmap 1.2 records a first attempt that failed because the rollouts were
generic prose, then re-anchored at restatement headers and verified firing
(findings 0.6). So the correction is narrower than first written -- masked150 was
teacher-forced, not "on-policy was never tried".

**Compounding factor.** `divergence = reverse`: mode-seeking, so it sharpens the
output distribution. A sharpened model with no recovery behaviour collapses onto
a repeating mode and stays there, even sampling at T=0.7.

**What the evidence rules out**, after chasing each one:
  * NOT the KV/recurrent cache -- cached vs full recomputation is identical at
    both 80 and 751-token contexts;
  * NOT thinking left on -- 0 think tags in 421 generations;
  * NOT the sampler -- T=0.7/0.8/20 is Qwen3's own non-thinking setting;
  * NOT MBPP ambiguity -- the base model sees the same underspecified prompts
    and does not get stuck. Ambiguity is the TRIGGER (it induces drift earliest),
    not the cause;
  * NOT the architecture (GDN-2 / MLA d_c=512 / NoPE / VeRA) -- though a
    surgery-only arm is still the clean separator and is queued. Note GDN-2 is an
    upgrade to linear-attention layers the Qwen3.5 family already has (the 27B
    config carries linear_key_head_dim, linear_num_value_heads, mamba_ssm_dtype;
    the 35B-A3B GGUF has ssm_* on 30 of 40 blocks), not a foreign graft.

**Diff supervision: D10.2 already settled this.** Human commit diffs are NOT
distillation targets -- the teacher may assign them low probability -- so they
belong in the CE data term, weighted and monitored separately. D12 adds that the
diff tier therefore needs no divergence term at all and is free on both axes. The
measurement below was re-derived without consulting either. 50.1% of supervised tokens
are raw unified diffs (925K of 1.84M; the "100%" figure reported earlier came
from `n_target_tokens`, a metadata field only diff episodes carry -- it was
wrong). Diffs are long, structured and machine-generated, so they give drift
more room to accumulate, and their "block, then the block again with changes"
shape matches the REVISING bucket (14% of truncations). It does not explain the
stuck bucket, which repeats identically.

**Harness bug found alongside, independent of all this.** The MBPP prompt ends
with a CLOSED ```python fence containing an assert, immediately before the
assistant turn -- a copy attractor. mbpp/253 echoes that assert 34 times; 5 of
28 stuck generations echo it verbatim.

**Fix.** Retrain with `--on-policy` non-zero (GKD). 150 steps is cheap enough to
redo. On-policy steps cost a 256-token generation each, so wall-clock rises;
that is the price of closing the loop.

**Method note.** Three detectors gave 1.4%, 71% and 12.4% for "how degenerate is
this", and the first two were reported as fact before anyone read a generation.
A metric that can only be checked against another metric is not evidence yet.

## 17. Derive a run command from the LAST run's log, not from the change (2026-09-24)

Building the hidden-objective arm, four flags were dropped from the reference
configuration in four separate attempts, each time by composing a command around
the NEW feature instead of starting from what the previous run actually did:

  * `--vera-all 1024` omitted -> fell through to LoRA, which this project does
    not use. 51.71 M trainable instead of 34.24 M, and the arm would not have
    been comparable to anything.
  * `--doc-aware` omitted -> `--synth-data` is only read under it, so the
    synthetic corpus was silently dropped and the log never mentioned it.
  * `--synth-data` omitted -> on-policy anchors at restatement headers, which
    exist only in that corpus, so `--on-policy 0.5` fired ZERO rollouts while
    the run completed and reported on-policy as enabled.
  * `--grad-checkpoint` omitted -> peak memory 55.4 GiB against masked150's
    26.8, and the run OOM'd. This was first diagnosed as "the head swap
    materialises two V-wide tensors per chunk", which was plausible, specific
    and wrong.

Each failure presented as a property of the new objective rather than a missing
argument, and the last one produced a confident mechanical explanation for
something that was just a dropped flag.

A FIFTH followed, and it breaks the obvious version of this rule:

  * `--ckpt-above 0` omitted -> it DEFAULTS TO 8192 and means "only checkpoint
    windows LONGER than this", so at seq 8192 the condition is never true and
    gradient checkpointing never engages. Measured with instrumentation around
    the student forward:

        grad_ckpt=False   delta 33.92 GiB   peak 40.18
        grad_ckpt=True    delta  0.47 GiB   peak 11.22      (same 4,384 tokens)

    72x. Four runs OOM'd on this, and it was misdiagnosed twice -- first as "the
    head swap materialises two V-wide tensors per chunk", then as allocator
    fragmentation. Both were specific, plausible and wrong. A bisect
    (on-policy off, then head swap off, then hidden objective off) cleared every
    new component before instrumentation found the cause in one line.

**The rule is: diff against the recorded ARGS, not the log.** The log prints
"gradient checkpointing enabled" at setup, which is true at setup and then
reversed by a per-step length-conditional toggle on every window. Reading the
log would have confirmed the wrong thing. `logs/recovery-<tag>.json` carries an
`args` block with every flag as the run actually received it -- `ckpt_above: 0`,
`vera_all: 1024`, `grad_checkpoint: true` -- and that is the only faithful
record. Prose written by the program about its own configuration can be stale,
conditional, or reversed later; the args cannot.

The first three are now startup errors (--on-policy without --synth-data,
--synth-data without --doc-aware).

Corollary worth its own line: when a symptom appears only in the NEW
configuration, the instinct is to explain it with the new machinery, and that
instinct produced two confident mechanical explanations here before anyone
measured anything. Bisect first, instrument second, theorise last.

## 18. The model was never trained to STOP (2026-09-24)

The recovered model fails to terminate on 14.6% of generations where the
unmodified base model fails on 0%. Measured, 48 prompts, seeded, batched:

    healthy generations   median 252 tokens, MAXIMUM 637
    failures              7 of 48, every one running to the cap --
                          the SAME 7 at a 2048 cap and at a 6144 cap

Completely bimodal: a generation either finishes in a few hundred tokens or
never finishes at any budget. Reading them, the failure is a deliberation loop
in COMMENTS -- "# This is still complex. Let's try a different approach:"
followed by a restatement of the same approach, four of the seven dominated by
comment lines. Semantic repetition, not verbatim, which is why token-periodicity
scored it 1.4% and duplicate-lines scored it 71% on the same generations.

**Cause, and it is mechanical.** The recovery run supervises the end-of-turn
token exactly zero times:

  * 70% of steps draw from fineweb_edu_long, which contains 0 EOS / <|im_end|> /
    <|endoftext|> tokens in 1.38M measured. Raw text teaches continuation only.
  * the other 30% are episodes, where `ASSISTANT_SPAN` captured
    `(.*?)(?:<\|im_end\|>|\Z)` -- group(1) stops BEFORE the terminator, so the
    assistant's content was supervised and the token that ENDS the turn was not.
    Verified: 0 of the assistant-closing <|im_end|> tokens carried a mask.

So the base model's instruction-tuned stopping behaviour had no gradient
maintaining it and decayed. This explains why both arms show the defect equally
(identical masking), and why on-policy did not touch it -- at 6.7% effective
exposure it was never the mechanism at issue. D16 attributed the degeneration to
exposure bias; that may still contribute, but it is not required to explain a
model that was never once shown how to stop.

FIXED: the terminator is now inside the captured group (verified 20/20
assistant-closing tokens supervised across 20 episodes).

SECOND, SUBTLER BUG, fixed alongside: EpisodeDataset caches the MASKS, and the
cache key was (path, mtime, tokenizer). Changing how a mask is computed left
every existing cache valid, so the fix above would have been a silent no-op on
any machine that had already built one. The key now includes MASK_VERSION.

## 19. The stop hazard is zero exactly where the corpus has no terminators (2026-09-25)

#18 explained why the model never learned to stop. This is the quantitative
shape of what it learned instead, and it moves the fix from the objective to the
data.

An autoregressive LM with EOS in the vocabulary already IS a discrete-time
survival model, and this needs no assumption about the learned function -- only
two properties of our own code: the softmax sums to 1 over V, and the decode
loop stops iff EOS is sampled. Then P(stop at T) = h_T * prod_{s<T}(1 - h_s)
with h_t = p(EOS | x_<t) is the chain rule. The survival curve is a property of
the DECODER. What is NOT free, and is the entire empirical question, is the
SHAPE of h_t -- survival analysis normally earns its keep through structural
assumptions (proportional hazards, smooth baseline, monotonicity) and our h_t
has none of them guaranteed. So it was measured, not assumed.

Realised per-token hazard, from the 6144-cap generation dumps, against the
hazard implied by the corpus's own assistant-span lengths:

    bin           GEN control   GEN hidden   CORPUS terminators
    (  64, 128]     0.00284      0.00229        0.00013
    ( 128, 256]     0.00336      0.00267        0.00259
    ( 256, 512]     0.00335      0.00158        0.00238
    ( 512,1024]     0.00088      0.00036        0.00218
    (1024,2048]     0.00000      0.00000        1.00000
    (2048,4096]     0.00000      0.00000          --

CORRECTED 2026-09-25. The table above first carried 0.00015 for `hidden` in
(4096,6144], which was an ARTEFACT OF MY OWN ANALYSIS, not a measurement. The
dump's `n_tok` re-tokenises the decoded text rather than recording the generated
length, so four capped rows came back as 6138/6142/6143 against a 6144 cap and a
"stopped before the cap" test scored them as TERMINATIONS. Read against the
authoritative per-row truncation flag the value is 0.00000 like every other arm,
and `hidden`'s longest terminating generation is 625 tokens, not 6143.
stuck_rate now writes `truncated` and `max_new` into the dump; any termination
analysis must read `truncated` and must never compare n_tok to max_new.

Where the corpus has support (128-512) the model's hazard matches it to within a
factor of two, and is if anything HOTTER than the data -- it over-stops early.
Past 1024 tokens the hazard is not small, it is 0.00000, across 5120 further
tokens: an absorbing state. A constant-hazard model is badly wrong (censored MLE
predicts median 911 and P(len>1024)=0.458; observed 271 and 0.146), so this is a
hazard CLIFF, not a weak hazard.

The cause is data support. Over the full corpus -- 1973 episodes, 6041 assistant
spans, all 6041 now carrying the terminator (#18 verified corpus-wide):

    p50 401   p75 690   p90 1152   p99 1796   max 2794
    beyond 1024 tokens: 12.7% of spans;  beyond 2048: 4 spans (0.066%)

The top of the distribution TAPERS SMOOTHLY (176/87/43/9/4/3 spans in the bins
from 1024 to 2010) with no pile-up at 2048, so the ceiling is a property of the
source data, not a truncation bug. There is nothing to un-filter. The model has
essentially never seen an assistant turn longer than ~2000 tokens end, and our
benchmarks generate to 2048-6144.

CONSEQUENCES:

  * No architectural explanation is required. A NoPE/length-conditioned-hazard
    story was drafted and is withdrawn -- the hazard is zero exactly where the
    data is absent, which is sufficient. NoPE is not exonerated, just not needed.
  * Reparameterising the loss into survival form buys nothing: it is already
    that factorisation.
  * There IS a length bias worth fixing, and it points the opposite way to the
    length-normalisation literature. Our loss is a MEAN OVER POSITIONS, so each
    position carries weight 1/L. The single terminator in a 690-token span gets
    1/690; in a 240-token span, 1/240 -- the stop decision is weighted ~3x more
    in short spans, and the long spans are exactly the ones whose terminators we
    need. Normalisation SUPPRESSES the signal in the broken regime. The fix is
    to DE-normalise the terminator: weight it independently of L.
  * Long terminating spans must be SYNTHESISED (the teacher is the natural
    supplier and already serves completions), not recovered from the source.
  * A decode-time hazard floor would eliminate non-termination outright, since
    the hazard is literally 0. Symptom patch, but a complete one, and it should
    be kept separate from any claim about the model.

MEASUREMENT CAVEAT that applies to every arm comparison in this project: "Mind
the Cap" (2608.04160) finds length normalisation shifts results by up to 38.9
points and can REVERSE rankings where the generation cap binds. Ours binds on
14.6-31.2% of generations. Censored comparisons here are not reliable, which is
also why the swap arm's 17% at a 1024 cap is not comparable to control's 14.6%
at 6144 -- the matched-cap rerun is the only admissible comparison.

## 20. Evaluate the LAST checkpoint, not the ppl-selected "best" one (2026-09-25)

`-best.pt` is selected on ppl@8192. On swap-exactkl150 that rule picked step 100
over step 150 on a 0.005-nat difference (11.310 vs 11.315 -- noise), while step
150 was better on every closed-loop measure:

    step 100  ("best")   ppl 11.310   hit cap 21%   stuck 6%   novel-tail 0.80
    step 150  (last)     ppl 11.315   hit cap 17%   stuck 2%   novel-tail 0.89

So selection on an open-loop metric discarded the better-terminating model, which
is the defect the arm exists to fix. At 150 steps there is not enough training
for late-run overfitting, so the last checkpoint is very probably also the best
and the ppl tie-break is measuring noise.

RULE: pass ckpt/adapters-<tag>.pt to --stuck-ckpt and to any benchmark build,
not ckpt/adapters-<tag>-best.pt, and state which checkpoint a reported number
came from. Keep writing -best.pt, but do not read from it. The longer-term fix
is to break near-ties in ppl with a closed-loop term rather than to keep
selecting on ppl alone.

This is a specific instance of the standing warning that open-loop metrics (CE,
ppl, top-1) do not reliably predict closed-loop behaviour -- see #19's
measurement caveat and D16.

## 21. The terminator fix buys termination only INSIDE the corpus's support (2026-09-25)

The falsifiable prediction from #19 was run: swap-exactkl150 (terminator fix +
head swap + exact reverse KL + on-policy), final checkpoint, evaluated at the
same 6144 cap as the other two arms.

    arm       terminated   longest TERMINATING   hazard >1024   stuck   novel-tail
    control      41/48            637              0.00000      14.6%     0.767
    hidden       33/48            625              0.00000      14.6%     0.671
    swap         38/48            738              0.00000      12.5%     0.755

PREDICTION CONFIRMED, and it is a negative result for the fix as a cure. At a
1024 cap the terminator fix took non-termination from 83% to 17% across training
-- real, monotone, and reproducible. At 6144 it is indistinguishable from the
control: `stuck` 12.5% against 14.6% is 6/48 against 7/48, ONE generation, and
the arm actually terminates LESS often overall (38/48 against 41/48).

The hazard past 1024 tokens is still exactly zero. Across 144 generations in
three arms there is not one that terminates beyond 738 tokens. The distribution
is perfectly bimodal: end before ~740 tokens, or never.

So supervising the terminator restored the stop decision WHERE THE CORPUS HAS
EXAMPLES OF IT and nowhere else, which is what #19 predicts and is the cleanest
available evidence that the binding constraint is data support rather than the
objective, the architecture, or exposure bias. The mask bug was real and worth
fixing; it was not the whole defect.

WHAT THIS LICENSES: long terminating assistant spans are now the top item, and
the teacher can supply them -- probed uncensored, it terminated 6/6 with lengths
4759/4778/4954/4974/5322/6389, every one beyond the corpus maximum of 2794. At
one server slot and ~21 tok/s that is ~20 h for 300 spans, so instruction
backtranslation (arXiv:2308.06259) over documents we already hold is the cheap
route: generate a ~50-token instruction for an existing long document instead of
a ~5000-token answer, a 100x reduction. Register differs from technical answers,
so a mix is likely needed.

CONFOUND, stated: the swap arm differs from the control in the head swap and the
divergence as well as the terminator fix, so this isolates neither. It bounds the
terminator fix's reach, which is what was being tested.

## 22. The 15-point HumanEval gap is NOT a training-configuration problem (2026-09-25)

Five arms, measured on the FULL HumanEval at identical settings (--no-think,
max_new 768, batch 64), against the model they were carved from:

    arm                     HumanEval   ppl@8192   ppl@2048
    original (base NF4)       79.9%        --         --
    masked150                 67.7%        --         --
    control-revkl150          65.9%       9.653       --
    recipe-moe600             64.6%       9.347      7.453
    recipe-moe600-f2a2        65.9%       9.339      7.412

Everything below the base is ONE POPULATION. At n=164 the binomial sigma is ~6
items, and the whole 64.6-67.7% spread is 5 items wide. Spanning that band:
three recipe configurations, a 4x change in training steps (150 -> 600), AdamW8bit
vs NAdamW, EMA weight averaging on vs off, and a novel attention mechanism. None
of it separates.

WHAT THIS KILLS. The working assumption all through 2026-09-25 was that the
deficit was recoverable by training: better recipe, more steps, better optimizer.
Every one of those was tried and the number did not move. Specifically:

  * RESTORING THE RECIPE did not help. masked150's args were diffed against the
    newer arms and four settings had been silently dropped: --vera-lr 0.01 (so
    adapters trained at the DENSE rate, a 50x smaller displacement ceiling --
    lr*steps 1.5 vs 0.03), --train-norms (103 fewer trainable tensors including
    the 64 ScaleNorm gains), lr 6.7x too high, and --episode-frac cut 0.5 -> 0.3.
    Restoring all four took ppl@8192 11.037 -> 9.347 and hit-cap 19% -> 4%, and
    HumanEval went 65.9% -> 64.6%, i.e. nowhere.
  * MORE STEPS did not help, and stopped paying at ~200. The control's ppl@8192
    fell 1.399 over steps 0-100 and 0.19 over 200-500, bottoming at step 500 and
    RISING by 600. That is at 8% of one epoch over the 42.8M-token corpus, so the
    earlier "we use 2.9% of the data, we must be step-limited" arithmetic was
    correct and the inference from it was wrong: availability is not consumption.
  * F2A2 did not help. See #23.

WHAT PERPLEXITY IS WORTH AT THIS SCALE: nothing. 0.3 nats of ppl@8192
(9.653 -> 9.347) bought 0 HumanEval points. Compare the swap arm, where 1.8 nats
mapped to 35 points. So ppl ranks arms when the gap is large and is pure noise
below ~0.5 nats. Reporting "0.31 nats better" as progress, which happened
repeatedly during this session, was measuring something that does not cash out.

WHAT DID MOVE, and it is the one honest win: TERMINATION. hit-cap at a 1024 cap
went 19% -> 4%, from the #18 mask fix plus #19's corpus finding. It did not show
up in HumanEval because HumanEval at max_new 768 is not where non-termination
costs points -- see #21's caveat about censored comparisons.

WHERE TO LOOK INSTEAD. If 15 points are lost in the SURGERY rather than in
training, no recipe recovers them. The untested decomposition is to ablate each
conversion against the unconverted base ON HUMANEVAL, not on perplexity:
GDN-2 lift (24 layers), MLA compression at d_c=512 (8 layers), NoPE, ScaleNorm,
per-head-q. Each alone, each cumulative. That is a day of benchmark runs and it
is the only measurement that would say where the capability actually goes.

## 23. F2A2: the mechanism works, the model keeps ~18% of it, it buys nothing (2026-09-25)

Roadmap 3.2, trained as a controlled pair against recipe-moe600 differing by two
flags (--f2a2 --f2a2-lr 1e-3). Result:

    ppl@8192   9.347 -> 9.339   (-0.008)
    ppl@2048   7.453 -> 7.412   (-0.041, consistent across all four tau=1 evals)
    HumanEval  64.6% -> 65.9%   (106 -> 108 items, sigma ~6: noise)
    alpha_diag 0.8173           (default at W=I is 0.42, so ~18% borrowing kept)
    cost       524,416 params on 8 of 32 layers

The verdict is a NULL, but an informative one, because three things were verified
rather than assumed:

  1. REACHABLE. Its own LR group at 1e-3 gives a displacement ceiling of
     lr*steps = 0.6. In the dense group at 3e-5 the ceiling is 0.018 and the
     mechanism could not have moved -- the same failure as the up_k widening
     (roadmap 1.4) and the head-swap projection before --proj-lr.
  2. ACTUALLY USED, not merely installed. alpha_diag went 0.646 -> 0.762 -> 0.815
     -> 0.830 -> 0.817 across the tau=1 evals, i.e. the model moved it from the
     0.42 default and SETTLED at ~0.82 rather than driving it to 1. Probed
     directly: resetting W=I and head_scale=0 on the step-200 checkpoint moved
     alpha_diag 0.6053 -> 0.4202, so +0.185 of it was learned, not default.
  3. ACTUALLY ON during evaluation. tau is a NON-PERSISTENT buffer, so every
     rebuild starts it at 0 = maximal mask = alpha exactly the identity. The
     training loop ramps it; build() and --stuck-only skip the loop. The first
     benchmark ran with F2A2 fully masked OFF and would have reported "no effect"
     while measuring its ABSENCE. Both paths now set tau=1 explicitly. This is
     the single most dangerous bug of the session: a disabled mechanism and an
     ineffective one produce identical numbers.

DESIGN HISTORY, because three formulations failed and each failure was specific:
a learnable diagonal bias (approximate identity, pinned under Adam's eps at 24,
5.8e-02 perturbation at 6); a zero-init signed gate on the output blend (s_h went
NEGATIVE, extrapolating outside the convex hull -- and clamping it creates a dead
parameter); an annealed mask with FIXED magnitude (out-shoutable, alpha_diag 0.46
instead of 1). The version that works masks the off-diagonal by
(1-tau)(30 + 2*max|S|): convexity is the shape of a softmax rather than an
invariant to police, and the init is exact for any weights.

SCOPE CEILING: F2A2 only applies to the 8 softmax layers. The other 24 are GDN-2
linear attention with no softmax heads to mix, so even a clear win is bounded to
a quarter of the network.

## 24. Surgery ablation, and a prediction recorded BEFORE the result (2026-09-26)

#22 showed the ~15-point HumanEval gap survives every training-side lever, so the
deficit is in the surgery. Four arms at 150 steps, each changing ONE thing against
ablate-base150 (the restored recipe). 150 steps because #22 measured HumanEval as
flat between 150 and 600.

    arm              MLA allocation                      dial
    base150          water-filled 4096                   nope     reference
    uniform150       UNIFORM 4096, 512/layer             nope     POLICY
    mla8094          water-filled 8094 (ranks doubled)   nope     BUDGET
    rope150          water-filled 4096                   c0       NoPE

THE HYPOTHESIS THAT MOTIVATED uniform150, which is not one I had considered: the
water-filling minimises KV reconstruction error AT INIT, and that may be the wrong
objective for a model that then trains.

  * RANK IS A PERMANENT CEILING. Measured spread of the heterogeneous plan, each
    group as a fraction of its joint maximum (2 * n_heads * head_dim):

        layer  3 heads [0,1,2]   106/1536 =  6.9%   <- starved
        layer  7 heads [0,1,2,3] 188/2048 =  9.2%
        layer 11 heads [0,1,2,3] 286/2048 = 14.0%
        layer 15 heads [0,1,2]   276/1536 = 18.0%
        layer 31 heads [1,2,3]   301/1536 = 19.6%
        layer 27 heads [0,1,2,3] 598/2048 = 29.2%
        layer 23 heads [0,1,2,3] 745/2048 = 36.4%
        layer 19 heads [0,1,2,3] 789/2048 = 38.5%
        layer 15 heads [3]       249/512  = 48.6%
        layer  3 heads [3]       253/512  = 49.4%
        layer 31 heads [0]       305/512  = 59.6%   <- favoured

    An 8.6x spread. The 6.9% group can never represent more than 106 dims of KV
    subspace however long it trains. Low init error means the ORIGINAL model's KV
    was already near-low-rank there; it says nothing about the capacity that group
    needs in order to ADAPT.

  * THE STATISTICS DESCRIBE A FUNCTION THAT NO LONGER EXISTS. The covariances come
    from the pre-conversion model, but GDN-2 replaced 24 layers with linear
    attention and NoPE removed positional encoding, so what each surviving softmax
    layer has to do changed after they were measured.

NOTE mla8094 CANNOT test this: doubling the ranks preserves the heterogeneity and
its extremes, so it would return negative even if allocation policy is the culprit.
The budget and the policy are separate questions and need separate arms.

PREDICTION, written before any arm finished: if allocation policy is the problem,
uniform150 beats base150 on HumanEval DESPITE worse reconstruction error at init,
and mla8094 does not. If the problem is total capacity, mla8094 beats base150 and
uniform150 does not. If NoPE is the problem, rope150 beats base150 and the other
two do not. If all four land inside the 64-68% band that every arm has occupied so
far, then none of MLA allocation, MLA budget, or NoPE is where the 15 points go,
and the remaining suspects are the GDN-2 lift on 24 layers and 4B capacity itself.

TWO REBUILD BUGS FIXED BEFORE LAUNCHING, both of which would have invalidated the
comparison silently:
  * bench_full took ONE --mla-groups for every arm it is given. Rebuilding mla8094
    with the 4096 file gives differently sized latents, strict=False drops them,
    and the arm evaluates with RANDOMLY INITIALISED latents and no error.
  * build() hardcoded install_rope_dial(m, 0, "global"), i.e. always NoPE, so
    rope150 would have been evaluated as a NoPE model. The dial changes no tensor
    shapes, so nothing would have complained. build() now takes `dial` and
    bench_full passes it.

## 25. Allocation washes out, capacity does not (2026-09-26)

The surgery ablation of #24, perplexity results. Four arms at 150 steps, each
changing ONE thing against ablate-base150 (the restored recipe):

    arm          init ppl@8192   final    init delta   final delta   SURVIVES
    base150         11.037       9.540       --           --           --
    uniform150      11.837       9.602      +0.800       +0.062         8%
    mla8094         10.682       9.282      -0.355       -0.258        73%
    rope150         10.472       9.172      -0.565       -0.368        65%

THE ASYMMETRY IS THE FINDING, and it was predicted from first principles before
the arms ran (the hypothesis was not mine -- see the motivation in #24):

  * REDISTRIBUTING a fixed budget is something training largely UNDOES. Uniform
    allocation starts 0.800 nats behind and finishes 0.062 behind: 92% of the
    handicap is recovered. Water-filling buys a real 0.800 nats at init and keeps
    0.062 of it.
  * REMOVING CAPACITY OR INFORMATION is not recoverable. Doubling MLA rank keeps
    73% of its init advantage; keeping RoPE instead of NoPE keeps 65%.

So the CARE water-filling spends its entire optimisation on the one axis that
adaptation neutralises, while holding fixed the axis that adaptation cannot touch.
The mechanism is exactly what #24 argued: rank is a PERMANENT ceiling, allocation
is only a starting condition, and reconstruction error at init cannot see the
difference.

WHAT ACTUALLY COSTS PERPLEXITY, after training:

    NoPE                       0.368 nats
    MLA rank at 25% vs 49%     0.258 nats
    MLA allocation policy      0.062 nats   <- negligible

CONSEQUENCE FOR THE PLAN: stop tuning the allocation. If MLA is to be cheaper it
should be cheaper in TOTAL RANK, and the interesting question is what fraction of
uncompressed is actually needed -- the current 25% costs 0.258 nats against 49%.

CAVEAT ON BOTH WINNERS: they buy their gains by giving back what the surgery was
for. rope150 abandons NoPE; mla8094 doubles the KV cache. They are different
points on the compression/quality trade-off, not free improvements.

AND THE OPEN QUESTION: #22 measured 0.3 nats buying ZERO HumanEval points, so
0.368 may reach nothing. The benchmarks decide whether any of this touches
capability or whether the ~15-point gap lives in the GDN-2 lift (24 of 32 layers,
three times MLA's surface) or in 4B capacity.

## 26. HumanEval at n=164 cannot resolve these arms; use GSM8K (2026-09-26)

Every arm comparison on 2026-09-26 was quoted against sigma ~6 items, which is the
standard error of a SINGLE arm. The standard error of a DIFFERENCE between two
independent arms is sqrt(2) times that: at n=164, p~0.70, it is 8.3 items = 5.1
points. Re-read with the right sigma, the entire HumanEval ablation resolved
nothing:

    comparison                    items   sigma_diff   significance
    mla8094   vs base150            4        8.3          0.5
    rope150   vs base150            2        8.3          0.2
    uniform150 vs base150           4        8.3          0.5
    base (79.9%) vs mla8094        11        7.7          1.4

NOT ONE PAIR, including the unconverted base against the best arm, was
distinguishable. Detecting a 5-point difference at 80% power needs roughly 1100
items per arm. HumanEval can see the swap arm's 18-point collapse (#21) and
nothing finer.

GSM8K has 1319 items. First result there:

    original (base)   90.5%   (reference run)
    base150           81.1%   1070/1319, 104.7 min
    masked150         80.3%   (reference run)

So the gap to the base is 9.4 points at ~6 sigma -- MEASURED, for the first time.
And base150 vs masked150 is 11 items at ~0.6 sigma, i.e. the four restored flags
plus NAdamW plus the terminator fix bought nothing over the old recipe on the
metric that can see it. That is the same conclusion as #22 but now demonstrated
rather than inferred from an under-powered test.

TWO PRACTICAL RULES FROM THIS:
  * arm comparisons go on GSM8K (n=1319), not HumanEval (n=164). HumanEval is for
    catching collapses, not for ranking.
  * do NOT quote a partial GSM8K number. The running rate fell 87.7% (at 464) ->
    85.7% (848) -> 82.8% (1184) -> 81.1% (final): the later items are harder, so
    partial figures systematically overstate. Only full-set numbers compare.

## 27. Recovery buys termination, not reasoning; NoPE outweighs MLA capacity (2026-09-27)

Step-0 arms finally give the denominator. Every earlier comparison was trained-vs-trained
or trained-vs-base, so "recovery works but cannot close a surgery gap" was
indistinguishable from "recovery is barely doing anything". It is the second.

PAIRED throughout (McNemar, exact binomial). All arms run the same GSM8K 1319, so
only discordant pairs carry information; the unpaired sigma_diff of 1.56 points I
quoted before is the wrong yardstick and understates power by ~30%.

    arm                     step 0   150 steps   delta      z        p
    base150 (NoPE, r=4096)  75.4%     81.1%      +5.76   +4.68   3.4e-06
    mla8094 (NoPE, r=8094)  83.3%     85.9%      +2.58   +2.59     0.012
    rope150 (RoPE, r=4096)  86.4%     86.4%      +0.08   +0.08       1.0

Then split by whether the STEP-0 generation hit the 768 cap (a truncation scores as
a failure, so merely learning to stop gains points):

    arm        net    from step-0 TRUNCATED     from step-0 COMPLETED
    base150    +76    +62  (82% of net)         +14  (103 gain / 89 loss)
    mla8094    +34    +32  (94%)                 +2  ( 69 / 67)
    rope150     +1    +35                       -34  ( 43 / 77)

ON ITEMS THE MODEL ALREADY FINISHED, 150 STEPS OF RECOVERY DOES NOTHING: +14, +2,
-34. rope150 nets zero because recovery fixed 35 truncations and broke 34 answers
that already terminated. So recovery is a termination fix -- the same thing the
fixed EOS mask addresses -- and the arm ranking is set entirely at construction
time. Consequence: effort belongs in the surgery (rank, NoPE), not in recovery
length, optimizer or EMA, which is where much of it went.

DAMAGE AT STEP 0, before any training:
    NoPE, holding r=4096          -10.99 points   z=9.15   (base150 vs rope150)
    r 8094 -> 4096, holding NoPE   -7.96 points   z=6.91
    GDN-2 + MLA-4096 alone          -4.1 points            (90.5% base vs 86.4%)
NoPE is the LARGER insult, and it is the one #25's capacity argument does not cover.

LATENT EXTENSIONS BOTH HURT. gate150 and taps150 both 1046/1319 = 79.3%, each -1.82
against base150 (z ~ -1.5, p ~ 0.14; same sign and size, individually short of
significance). NOT the zero-init-does-not-move failure of #23: down_g reached rms
7.0e-3 against the on-path latent's 0.025 (28%), taps 19%, because --latent-ext-lr
1e-3 is 33x the latent rate. They travelled far from identity while nothing else
co-adapted, so at 150 steps the perturbation outweighs the capacity. Retest needs a
lower ext lr or more steps, not a different mechanism.

CAVEATS. Re-scoring puts step0-base150 at 994 where the bench logged 993 (one item,
an extraction tie). The step-0 arms take 138 truncations against base150's 53, so
"surgery damage" as measured bundles a termination component.

## 28. Recovery budgets in the conversion literature, and which ones apply to us (2026-09-27)

Asked whether our data mix and recovery length are simply too small. Answer: the
budget gap is real and enormous, but the framing needs two corrections.

### 28.1 We do NOT do attention replacement, so most of those budgets are irrelevant

Qwen3.5-4B is NATIVELY a 24 GatedDeltaNet + 8 full-attention hybrid (3:1, the
Qwen3-Next scheme), pretrained that way for 15T tokens. Our "GDN-2" surgery LIFTS
the existing GDN layers to channel-wise gates (+503.32 M params tiled from
in_proj_b, EXACT AT INIT). We delete no attention layer.

So the linearization budgets do not describe our situation:
    LoLCATs          40 M tokens   (feature maps only, base frozen)   no GSM8K
    MOHAWK (Phi-1.5)  3.0 B        2% of pretraining, 3 stages        no GSM8K
    Llamba-8B          12 B        0.08% of pretraining               no GSM8K
    SUPRA              20 B        ~1% of pretraining                 no GSM8K
    Mamba-in-Llama     20 B        3 stages (KD -> SFT -> DPO)        GSM8K yes
Mamba-in-Llama is the only one reporting GSM8K, and only for hybrids that RETAIN
attention: 67.85 at 50% retained, 40.64 at 25%, 26.91 at 12.5%, and the FULLY
converted variant's GSM8K is absent from their own table. We sit at the 25% ratio
and score 81-86%, because Qwen pretrained that ratio rather than us carving it out.

### 28.2 The budgets that DO apply are 6,700x and 22,000x ours

    our budget                              0.90 M tokens (150 steps, measured)
    TransMLA, d_c=512 / ~93% compression      6 B tokens  -> 6,700x
    DroPE, pretrained RoPE -> NoPE           20 B tokens  -> 22,000x
At 336 tok/s: 6B = 207 days, 20B = 1.9 years. No data-mix change closes that.

TransMLA detail (arXiv 2502.07864v5, Table 1 + App. E Table 3; repo moved to
MuLabPKU/TransArch): two-stage 5B @ lr 1e-4 bs 256 then 1B @ lr 2e-5 bs 64, seq
4096, FULL fine-tune under ZeRO-3, PLAIN LM CROSS-ENTROPY -- no distillation, no
teacher. Recovered 43.26 -> 58.68 against 59.85 original = 93% of the gap. Their
milder 68.75% row still needed 500M tokens.

DroPE detail (arXiv 2512.12167, Sakana): 20B tokens = 0.5-2% of pretraining, full
parameter, and QK-Norm had to be ADDED to keep training stable after RoPE removal.

### 28.3 Nobody measures reasoning recovery. At all.

NOT ONE of TransMLA, DroPE, MOHAWK, LoLCATs, Llamba, SUPRA or Hedgehog reports
GSM8K or MATH for a converted model. TransMLA DEFINES gsm8k in lighteval/tasks.py
but leaves it out of the launched eval command, and its repo to-do reads
"[ ] Fine-tune on R1 distillation datasets." DroPE reports only ARC/HellaSwag/
PIQA/WinoGrande -- single-token multiple choice, far less sensitive to positional
damage than generative CoT.

So our paired-McNemar GSM8K setup is MORE sensitive than anything published here.
We are not failing to reproduce a known result; we are measuring an axis the field
avoids. That reframes #27: the wall is not evidence our method is uniquely bad.

### 28.4 Damage decomposition, with the GDN-2 lift contributing exactly zero

Because the lift is exact at init, every step-0 point is MLA and NoPE:
    base, unmodified                          90.5%
    GDN-2 + MLA r=4096, RoPE kept             86.4%   MLA-4096   = -4.1
    the same with NoPE                        75.4%   NoPE      = -11.0
    NoPE with r=8094                          83.3%   2x rank   = +7.9
NoPE is 73% of the 15.1-point total, MLA rank 27%.

RANK AND NoPE INTERACT, which nothing had shown. Doubling rank buys 7.9 under
NoPE, while compression costs only 4.1 IN TOTAL under RoPE -- so rank is worth at
least 1.9x more once RoPE is gone. Mechanism: with no positional signal, position
must be inferred from content, so a low-rank content bottleneck destroys the only
remaining positional evidence. Additivity is impossible (it predicts 94.3%, above
the unmodified base), so the effects are sub-additive. step0-rope8094 fills the
missing cell; it needs NO training, so it costs one build plus one GSM8K pass.

### 28.5 Data: no clean third-party mathematics corpus exists, and the field has none either

Our own #11 verdict ("there is no clean third-party mathematics or reasoning
corpus") is confirmed from the outside: NONE of the eight surveyed papers uses a
cleanly-licensed math/reasoning dataset. Worse, teknium/OpenHermes-2.5 carries NO
license tag on HuggingFace and is load-bearing in TWO of them (Mamba-in-Llama's
SFT stage, Llamba's KD stage). TransMLA's own mix fails our rule on python-edu
(Stack v2, NOASSERTION) and StackOverflow (CC-BY-SA); its open-web-math at 8% is
tagged ODC-By but #11.2 already rejected it for redistributing CC-BY-SA Math
StackExchange -- our audit was correctly stricter than the card.

One lead: cosmopedia-v2 is 15% of TransMLA's mix, ODC-By, and generated by
Mixtral-8x7B-Instruct (APACHE-2.0) -- exactly the shape #11.3 allows. Under audit.

### 28.6 Regression found while checking this

--synth-data data/synth_recall_4b.txt (72 MB) and --on-policy 0.25 were live
through control-revkl150/swap-exactkl150, then dropped at masked150 -- and since
all 12 run scripts were generated from masked150's recorded args, NEITHER HAS RUN
SINCE, including recipe-moe600 and every ablation. This is #17 happening again.
NOTE it is a RECALL corpus derived from FineWeb-Edu, so restoring it addresses
retrieval, not the GSM8K gap. Not restored in the ablation family, which must stay
internally comparable; belongs in the next recipe run.

CORRECTION: I flagged the 68 nemotron-swe-v1 episodes as violating a "no NVIDIA"
rule. There is no such rule. data_policy.md:174 accepts nvidia/Nemotron-SWE-v1
(CC-BY-4.0, R2E-Gym synthesised statements, capped 1 M tokens/repo). The NVIDIA
rejections were provenance-based and specific. Nothing needs filtering.

## 29. MatryoshkaKV is the only paper that measures reasoning under KV compression (2026-09-27)

### 29.1 Reasoning degrades faster than commonsense -- but see the CORRECTION in #31.2

MatryoshkaKV (arXiv 2410.14731, Table 2), LLaMA2-7B, the ONLY GSM8K numbers in the
entire KV-compression / MLA-conversion literature:

    KV budget      GSM8K   retained      general zero-shot retained
    baseline       34.95      --                   --
    50%            31.77    90.9%                  --
    37.5%          26.91    77.0%              92.6-93.1%
    25%            16.38    46.9%                  --

CORRECTION (#31.2): these GSM8K numbers are measured AFTER LoRA SFT ON GSM8K, while
the 93% commonsense figure is zero-shot. The 3x ratio I drew from this conflated two
protocols and is WITHDRAWN. The within-protocol comparison still holds: at 37.5%
budget GSM8K keeps 77.0% while the SAME SFT 4-task average keeps 84.1%. So EVERY
"we recover 93% of the gap" claim in TransMLA / MHA2MLA / CARE / X-EcoMLA is
measured on the axis that barely moves. Our GSM8K wall is the expected behaviour
of this class of surgery, not a defect peculiar to our pipeline. This retroactively
justifies #26's decision to rank arms on GSM8K rather than perplexity or HumanEval.

### 29.2 It also resets the budget target from 6,700x to 220x

MatryoshkaKV's recovery is ~200 M tokens -- the smallest FULLY SPECIFIED budget in
the survey -- training ONLY the orthogonal projections (Cayley-parameterised U,
U=(I+Q)(I-Q)^-1), with a KD loss (KL + LM at 1:3) against the uncompressed model.
That is structurally the closest precedent to what we do: few trainable params plus
distillation, not TransMLA's full fine-tune under plain CE.

    ours                    0.90 M      87 min
    MatryoshkaKV          ~200 M       6.9 days at 336 tok/s   <- realistic target
    CARE / TransMLA heal    1-3 B      69-207 days
    X-EcoMLA              3.4-7 B
    MHA2MLA                 6-12 B     0.6-1% of pretraining

### 29.3 Diminishing returns start around 1B tokens, and INIT QUALITY sets the floor

CARE (arXiv 2603.17946, Table 2), Llama-3.1-8B-Instruct, rank 512, avg accuracy:
    CARE(E) init      57.04 (0B) -> 62.59 (1B) -> 63.27 (3B)
    TransMLA (wiki)   55.52      -> 61.36      -> 61.86
    Palu (SVD)        38.75      -> 47.58      -> 56.30   still < GQA 58.24 at 3B
Most of the value lands by 1B. But Palu's worse init is STILL 7 points behind CARE
at 3B tokens -- initialization does not wash out there. NOTE THE TENSION with #25,
which found allocation washed out 92%: allocation (water-filling vs uniform at
equal total rank) is a far subtler difference than Palu's plain SVD vs CARE's
covariance-aware whitening. Not a contradiction, but #25 should not be read as
"init never matters".

CARE itself is a TRAINING-FREE init; it reuses TransMLA's healing pipeline
unmodified, so it is not an independent recovery recipe.

### 29.4 Our loss is already at the right end of X-EcoMLA's CE/KL ablation

X-EcoMLA (arXiv 2503.11132, Table 12): baseline 52.77; CE=1,KL=0 -> 48.54 (worst);
CE=0,KL=1 -> 50.84; CE=1,KL=0.1 -> 50.98 (best). We run div_term + ce_beta*excess_CE
with ce_beta=1.0 and lm_weight=0.0 -- i.e. NO teacher-free CE, their CE=0/KL=1 row.
Moving to their optimum is worth ~+0.14 in their units. NOT A LEVER; do not spend a
run on it.

Their token ablation IS worth acting on: 3.4B -> 6.8B tokens bought +0.49, while a
larger teacher bought +1.18. Quote: "leveraging a stronger teacher model is
generally more efficient for improving accuracy than simply increasing training
data." We already run a 35B-A3B teacher for a 4B student (8.75x), so that lever is
already pulled and there is no cheap upgrade left on it.

### 29.5 Partial-RoPE recipe to copy, if we go that way

MHA2MLA (arXiv 2502.14837) keeps only the top-r rotary subspaces chosen by a
head-wise contribution score, and ablates the selection rule: S_high / S_low /
S_uniform / S_2-norm (Tables 3, 5, 7). Budget 12,000 steps, effective batch 256,
lr 1e-4, warmup 10% then 1-sqrt decay, FULL-parameter fine-tune, plain CE, no
distillation. Degradation at 68.75% KV reduction: -0.08% to -1.93% depending on
size. NO GSM8K anywhere, so its "-0.27%" cannot be read as a reasoning result.

### 29.6 Licensing: every disclosed recovery set in this family fails our rule

    tatsu-lab/alpaca            CARE calib, Palu LoRA, Eigen LoRA   CC-BY-NC-4.0  FAIL
    BAAI/Infinity-Instruct      X-EcoMLA SFT                        CC-BY-SA-4.0  FAIL
    teknium/OpenHermes-2.5      X-EcoMLA SFT (also Llamba KD)       no tag        FAIL
    bigcode/stackoverflow-clean MHA2MLA                             no tag        FAIL
    smollm-corpus core          MHA2MLA                             ODC-By        ok
    open-web-math               MHA2MLA                             ODC-By tag    FAIL per #11.2
There is no clean published recovery recipe we could replicate as-is. #11's verdict
stands and is now corroborated from outside the project.

## 30. Throughput correction: base150 is a slow outlier, not a structural baseline (2026-09-27)

I extrapolated every token-budget wall-clock from base150's 174 tok/s and inferred
that ungrouped latents were ~1.93x faster. BOTH WRONG. Median steady-state tok/s,
identical --seq 8192 and episode fraction:

    base150      174.2      <- the outlier
    mla8094      336.2      (MORE rank than base150)
    rope150      336.7
    taps150      336.3
    gate150      336.0
    uniform150   335.7
    ungrouped150 334.5

Grouping is not the cause: the water-filled plan is 1-2 groups/layer against
ungrouped's 1, and mla8094 has double the rank at full speed. base150 logged
"avail 23 GiB" against 40+ GiB for every other arm, so it ran under memory
pressure -- environmental, not architectural. Its GSM8K score is unaffected
(accuracy does not depend on throughput), so #27 stands.

CORRECTED BUDGETS at 336 tok/s:
    ours              0.90 M     45 min
    MatryoshkaKV      ~200 M     6.9 days     <- the reachable target
    CARE/TransMLA      1-3 B     34-103 days
    TransMLA d_c=512     6 B     207 days
    DroPE               20 B     1.9 years

LESSON: I generalised a rate from one run without checking the others, and built a
mechanism story ("fewer, larger matmuls") on top of it. The check cost one command.

## 31. MatryoshkaKV read at implementation level: what to copy, skip, build (2026-09-27)

Paper arXiv 2410.14731v2 (ICLR 2025) plus the released code,
github.com/The-kamisato/MatryoshkaKV-cache (single commit, NO LICENSE file).
Several load-bearing facts are in the code and NOT in the paper.

### 31.1 Absorption: WE HAVE A STRUCTURAL ADVANTAGE, and it constrains partial RoPE

They compress keys POST-RoPE. Appendix B states why: U^K sits between the rotation
and the query, so U^K cannot be folded into W_Q/W_K and q must be projected at every
decode step. That is the same obstruction that forces DeepSeek's decoupled RoPE head.

Under NoPE the obstruction vanishes, and more strongly than I had assumed: ANY
invertible change of latent basis is EXACTLY absorbable.
    c <- U^T c,  up_k <- up_k U
    ((up_k U)^T q)^T (U^T c) = q^T up_k U U^T c = q^T up_k c
Absorption is a basis-free statement about the bilinear form q^T up_k c; orthogonality
only makes truncation a projection rather than an oblique one.

CONSEQUENCE FOR PARTIAL RoPE: it must be done as a SEPARATE DECOUPLED HEAD, shared
across heads (DeepSeek's design, so the extra cache is one small head), NOT by
keeping RoPE on a subset of latent dimensions -- that would forfeit absorption on
those dimensions and reintroduce exactly the cost MatryoshkaKV had to eat.

Value side ports cleanly and they already do it: U^V folded into W_O, truncation =
dropping columns of W^OV (merge_wights(), modeling_pcallama_trial.py:341-351).

### 31.2 CORRECTION to #29.1: their GSM8K is a task-SFT number

Table 2's caption says "after SFT"; its 100% row (GSM8K 34.95) is LLaMA2-7B-base +
LoRA SFT ON GSM8K. Proof: the same table reports HellaSwag 93.94 where Table 1's
zero-shot base is 74.00. There is NO GSM8K evaluation of the CPT-only model anywhere
in the paper. So my "reasoning degrades 3x faster than commonsense" (#29.1) compared
a task-SFT'd generative benchmark against zero-shot multiple choice -- WITHDRAWN.

What survives, same models and same protocol:
    budget   CPT 6-task zero-shot   SFT 4-task avg   SFT GSM8K
    37.5%          93.1%                84.1%          77.0%
    25%            79.7%                68.5%          46.9%
GSM8K degrades faster than PIQA/HellaSwag/OBQA measured the same way. That is enough
to justify #26 (rank arms on GSM8K), but the clean 3x figure was mine, not theirs.

Noise: the GSM8K column is NON-MONOTONIC (87.5% = 35.25 > 100% = 34.95; 50% = 31.77 >
62.5% = 31.46) on a 34.95 baseline over 1319 items, so ~+-1.5 points. "90% retention
at 50%" is really 85-95%.

### 31.3 Matryoshka multi-rank sampling helps AT A FIXED RANK. Copy this.

Fig. 4(Right), their "w/o Matryoshka" ablation: a model trained at a FIXED 50% budget
scores ~0.815 relative GSM8K at 50%, while the multi-rank-trained model scores ~0.90
AT THE SAME 50%. The multi-rank model beats the single-rank model at the single rank,
so it is a REGULARISER, not only an adaptivity mechanism. (The paper never says this;
it only notes the fixed model fails to generalise to other budgets. Read off the
figure, values not tabulated. Confound: LoRA is co-trained, one seed, one task.)

Implementation is free: ONE rank drawn per micro-batch, ONE forward pass.
    trainer.py:447-449:  16 * torch.randint(1, 9, (32, 32))
i.i.d. uniform over {d/8..d}, independently per (layer, head) AND per K/V -- 2048
draws, Monte-Carlo over steps, NOT a sum over ranks inside a step. The full rank MUST
be in the schedule. Appendix G: results are insensitive to the schedule granularity.

At 0.9M tokens, where overfitting the distillation set is a live risk, a structured-
dropout regulariser should help proportionally MORE than at their 200M. Cheap test:
fixed d_c=512 vs sampling d_c in {128,256,384,512} per layer, both evaluated at 512.

### 31.4 Things the paper omits that the code reveals

  * TRAINABLE MEAN VECTORS mu_K, mu_V per (layer, head), re-added exactly. Absent from
    the paper entirely. Under NoPE q^T mu_K is constant over keys and CANCELS in the
    softmax, so the key mean is a no-op for us; the VALUE mean is a genuine free
    post-attention bias worth keeping. (Algebra straightforward; verify numerically.)
  * "Only U is updated" (S4.1) is CONTRADICTED by the released script: it does not
    override freeze_trainable_layers (default 2), so layers 30-31 train FULLY at
    5e-5, ~400M params. Part of their recovery is base-weight adaptation. LESSON:
    assert the trainable set explicitly, never trust a freeze flag.
  * Loss direction resolved: KD=0.25, LM=0.75 (trainer.py:443) -- LM DOMINATES 3:1.
    Forward KL, T=1.0, full vocab, mean per token. Teacher = a separate frozen copy
    of the original model. The stage-2 SFT has NO KD term at all.
  * LR: projections 1e-3, everything else 5e-5 -- a 20x RATIO. Relevant to #27: our
    --latent-ext-lr 1e-3 against a 3e-5 dense rate is 33x, which the user judged too
    high. 20x of our dense rate is 6e-4, and that is the validated number to retest at.
  * bf16 ROUND-TRIP IS NOT FREE: their full-rank row loses HellaSwag 74.00 -> 72.05
    and PIQA 78.50 -> 76.66 purely to the (x-mu) -> U -> U^T -> +mu path in bf16.
    Keep down/up-projection accumulation in fp32. (We already hold new params fp32.)
  * No token-budget ablation exists. The data-efficiency claim is asserted, never
    measured, so the paper says NOTHING about budgets below 200M tokens.

### 31.5 Skip

  * The Cayley/orthogonality machinery. Their own S5.4: non-orthogonal is "comparable
    ... when the cache budget is less than 50%"; orthogonality only preserves the
    FULL-RANK identity, which we do not ship. Costs a batched linalg.solve every
    forward pass.
  * The greedy rank search: O(L*H*2) forward passes per round, objective = final
    hidden-state Frobenius error on 32 concatenated calibration samples, and it buys
    the TRAINED model only +0.3 to +1.9 points (vs +9 to +25 for untrained PCA).
    Our water-filling allocator is cheaper. But KEEP their priors as a sanity check:
    shallow layers need more rank; only a minority of deep heads need high rank; and
    at a 37.5% budget the search gave KEYS 32.28% vs VALUES 42.72%.
  * Post-RoPE key compression -- irrelevant to us, see 31.1.

### 31.6 Build

  * Rank sampling over a SHARED grouped latent, not per-head 128-d blocks. Their
    (32,32) rank tensors are hard-coded; ours must be per (layer, group).
  * Matryoshka x VeRA interaction. They needed TWO STAGES (LoRA first, then joint)
    because task gradients misalign the adapter (Appendix C). At 0.9M tokens a split
    is expensive -- prefer biasing the rank schedule toward full rank for the first
    10-20% of steps.
  * A token-budget ablation, since nobody has one.

## 32. First model line: GDN-2 + MLA with the NATIVE partial RoPE kept. NoPE deferred. (2026-09-27)

### 32.1 The base model was already partial RoPE, which I had missed

ckpt/qwen3.5-4b-stageAB/config.json:
    head_dim 256, partial_rotary_factor 0.25  -> 64 rotary dims + 192 NoPE dims/head
    full_attention_interval 4                 -> 8 full-attention layers of 32
    linear_conv_kernel_dim 4                  -> the GDN short conv
64 is EXACTLY DeepSeek's qk_rope_head_dim. So our "NoPE" never converted a RoPE
model: it zeroed a 64-dim rotary slice in a design already 75% NoPE, one the
pretrained weights were optimised around. rope_dial.py's own docstring said so
("Qwen3.5 uses partial RoPE: 64 of head_dim 256 are rotary") and I framed the
ablation as full-RoPE removal anyway.

Dial semantics, for the record:
    nope -> keep 0,  policy global   (what base150/mla8094/taps150/gate150 ran)
    k4   -> keep 4,  local           (NEW; = MHA2MLA's r=4 default, 8 dims)
    k8   -> keep 8,  local           (NEW)
    c1   -> keep 16, local           (existed, NEVER benchmarked)
    k24  -> keep 24, local           (NEW)
    c0   -> keep 32, local           (native partial RoPE intact; rope150 ran this)
"local" keeps the FASTEST-rotating frequencies, which the literature says is the
correct end (p-RoPE 2410.06205: 0.75-RoPE 4.4414 Wiki PPL >= full RoPE 4.4627, NoPE
4.8594; HoPE 2410.21216; MHA2MLA S_high -0.82% vs S_low -5.25%). My earlier
hypothesis -- keep low frequencies for extrapolation -- was INVERTED.

### 32.2 DECISION: ship the native partial RoPE, defer NoPE

User: "Leave full nope for later, I think data and training steps is what's missing
actually, because a partial rope model should be easier to transform, not harder, but
for now the MLA compression and GDN2 upgrade are both good improvements for the first
model line."

The argument is sound. Pure NoPE is a validated design for EXACTLY this architecture
-- Kimi Linear (2510.26692) is KDA:MLA at 3:1 with kv_lora_rank 512 and NoPE on ALL
MLA layers, and its NoPE ablation BEATS its own RoPE variant at 128k (avg 54.5 vs
51.8, RULER 84.3 vs 78.8, MRCR 29.6 vs 22.0), delegating position entirely to the
KDA layers. But that is 5.7T tokens FROM SCRATCH (1.4T for the ablation), and the
only post-hoc conversion study, DroPE (2512.12167), needs 2-20B tokens (0.5-2% of
pretraining). Our 0.90 M is 2-3 orders short, so #27's null is what DroPE's budget
table predicts -- not evidence about NoPE's ceiling either way.

Measured in-distribution cost of pure NoPE at scale is SMALL: -0.06 val loss, -0.94
MMLU, -1.58 HellaSwag at 8B/750B (2501.18795 Table 2). Our 11.0 GSM8K points is far
larger than any published NoPE penalty, which again points at unpaid recalibration.

FIRST MODEL LINE = GDN-2 lift + MLA + dial c0 (native partial RoPE). Both retained
changes are justified: the GDN-2 lift is exact at init and contributes zero of the
step-0 damage (#28.4), and MLA at r=4096 costs only 4.1 points with position intact.

### 32.3 Convolutions cannot supply position. The algebra kills it.

With a shared up-projection, a depthwise conv gives
    score(t,j) = sum_i (w_i (*) q~_t)^T c_{j-i}
which has NO dependence on t-j anywhere. Contrast RoPE, where the score is provably
q_t^T R_{j-t} W^UK c_j, an explicit function of t-j. A conv changes the CONTENT of
the key, never makes the score a function of relative distance.

Also confirms the rank point: K_j = sum_i W^UK diag(w_i) c_{j-i} means every lag's
matrix shares range(W^UK), so the key-subspace ceiling is unchanged -- distinct from
multi-tap's independent A_i spanning up to (k+1)r. Conv is not a capacity mechanism
and not a positional one.

What a conv could give is a boundary anchor from left zero-padding (Islam ICLR 2020;
CPVT 72.4 -> 70.5 without) and local shift-compare (Based) -- and our 24 GDN layers
already supply strictly more, being unbounded-range gated recurrences. Jamba
(2403.19887): "with the Mamba layer, positional embeddings or mechanisms like RoPE
are not necessary"; Kimi Linear: KDA is "arguably stronger than ... short
convolutions or SWA". Every conv-as-PE success uses a much larger kernel (wav2vec 2.0
uses 128; ours is 4) or wants translation invariance. NOT PURSUED.

### 32.4 What the priority becomes

Data and steps, per 32.2. The concrete deficiencies, all already identified:
  * --synth-data regression: dropped at masked150, absent from all 12 scripts (#28.6)
  * the text corpus is LENGTH-filtered FineWeb-Edu with NO retained score metadata,
    not quality-score filtered (data_policy 12.1)
  * zero mathematics, while we evaluate on GSM8K (data_policy 12.2 settles the fix)
  * 0.90 M tokens per run against MatryoshkaKV's 200 M (#30)
  * Matryoshka rank sampling is a FREE regulariser that helps at a fixed rank (#31.3)
step0-rope8094 stays queued: with position intact it asks whether rank 8094 beats
4096, i.e. whether we can ship the smaller cache.

## 33. Architecture arms re-run under c0; two bugs invalidated the gate arm (2026-09-27)

Long run CANCELLED on request: settle the architecture first.

### 33.1 Every 150-step ablation except rope150 ran under --dial nope

base150, uniform150, mla8094, taps150, gate150, ungrouped150 -- all nope. #32.2 makes
line 1 c0, and #28.4 measured a strong interaction (rank is worth >=1.9x more once
RoPE is gone), so those arms settled components in a regime we have abandoned.
All of them are re-queued under c0, against ablate-rope150 as the control (GSM8K
1140/1319 = 86.4%; paired sigma on a difference ~1.4 points at n=1319, so treat
anything under ~3 points as unresolved).

### 33.2 TWO bugs made gate150 not test what it claimed

(a) --latent-ext-lr was 1e-3 = 33x the 3e-5 dense rate. gate150 and taps150 both hit
    79.3% against base150's 81.1%, with down_g reaching 28% of the on-path latent RMS
    in 150 steps -- travelling far from identity while nothing co-adapted. Default is
    now 6e-4 = 20x, MatryoshkaKV's validated ratio (#31.4).
(b) NEW BUG: `_is_x = lambda n: ("down_g." in n or ".extra." in n)` OMITTED
    xatlu.alpha, so the xATLU expansion parameters fell into the DENSE group at 3e-5
    instead of the latent-ext group. Measured displacement confirms it: xatlu.alpha
    rms 3.9e-4 against down_g's 7.0e-3, an 18x gap matching the 33x LR gap.
    SO gate150 TESTED A BARELY-EXPANDED ARCTAN, not the signed expanded gate the
    design is about -- alpha ~ 0 means the factor never went below 0 and never
    exceeded 1, i.e. attenuation only, which is exactly the behaviour xATLU was
    chosen to avoid. Predicate now also covers `.conv.w`.

This is the SIXTH time a new parameter group has been mis-specified in this codebase
(F2A2's gate, the up_k widening, the head-swap projection, the MLA cross-group
blocks, _is_f matching every mlp.gate_proj, and now xatlu.alpha). The pattern is
always a name predicate written from memory. Assert the group contents against a
real checkpoint before launching.

### 33.3 MLA conv: the design I dismissed was not the one proposed

#32.3 rejected "a conv" on the strength of score(t,j) having no t-j dependence. That
analysis was of a SINGLE key-side conv feeding a shared up-projection. The proposal
is convs on BOTH the down and up paths, and the down-side one is materially
different:
    pre-down:  c_j = down(sum_a v_a (*) x_{j-a}) = sum_a down diag(v_a) x_{j-a}
The latent bottleneck now compresses a temporal WINDOW of the residual stream instead
of a single position. That is new information at fixed d_c, and it is the plausible
mechanism for "mixing temporal information might resolve better than pointwise".

    pre-up:    K_j = up(sum_b w_b (*) c_{j-b}) = sum_b up diag(w_b) c_{j-b}
Here the earlier point DOES stand: every lag's matrix shares range(up), so the rank
ceiling is unchanged. That is precisely why this is separate from --mla-taps, whose
independent full A_i span up to (k+1)r. Conv = temporal resolution at k*r params;
taps = capacity at k*r*out. Orthogonal axes, as the user said.

I ALSO HAD THE COST WRONG. I worried a pre-down conv needs a per-token history of
hidden states. It does not: conv state is a FIXED-SIZE rolling buffer per layer,
O(k*d) regardless of context length -- the same thing Qwen3.5's GDN layers already
carry (linear_conv_kernel_dim 4). Nothing scales with sequence length.

On position, the honest reading is unchanged and weaker than the capacity argument: a
conv is shift-equivariant, so it cannot manufacture a t-j dependence; what it gives
is local ORDER sensitivity plus a left-zero-padding boundary anchor (Islam ICLR 2020,
CPVT). Since line 1 keeps RoPE anyway, this arm is being run for temporal resolution,
not for position.

IMPLEMENTED in surgery/latent_ext.py: DepthwiseCausalConv (delta-init, so EXACT at
install -- note the contrast with the zero-init of the gate and taps, which are
corrections rather than replacements), ConvDown, ConvUp, install_mla_conv.
VERIFIED: max|y-x| = 0.0 at init; with a random filter, perturbing t=3 changes
t=3..6 and leaves t<3 at exactly 0.0, i.e. reach exactly k and no backward leakage.
Replay detection added to retrieval_ab.py, keyed per side on `.down.conv.w` and
`.up_k.conv.w`, because ConvDown/ConvUp RENAME what they wrap like the others do.
Conv and gate/taps COLLIDE on the wrapped attribute name, so those arms are mutually
exclusive until someone writes a combined wrapper.

## 34. Mixture of Latents: novel, verified, queued (2026-09-27)

Per-token top-1 switch over the MLA latent DECODERS. Shared encoder, E alternative
up_k/up_v, cache = c_j plus one index (2 bits at E=4). src/mercurius/surgery/mol.py.

### 34.1 Prior art: NOT FOUND (arXiv + ar5iv, through Sept 2026)

Every published MoE-in-attention routes the QUERY/OUTPUT side and deliberately keeps
K/V shared, so the cache is untouched: MoA 2210.05144 ("the expensive matrix
projection of key sequence KW^k and value sequence VW^v can be pre-computed and
shared for all attention experts"), MoH 2410.11842, SwitchHead 2312.07987 ("keeping a
single, head-specific copy of the query and key projections ... is beneficial"),
JetMoE 2404.07413 ("cache size remains unchanged"). The reason they all avoid the K/V
side is that a cached key would be reused by queries routed elsewhere. Routing the
DECODER is tractable only because MLA's absorption moves the E x cost to the query
side PER DECODE STEP rather than per cached key.
Nearest structures: HydraLoRA 2404.19245 (one shared down A, several up B_i -- our
exact shape, but a LoRA adapter, soft-mixed, no cache); MatryoshkaKV varies the RANK
of ONE basis, not the basis (different mechanism, do not conflate).

### 34.2 Verified numerically, not asserted

  decode regrouping == naive value path   max err 7.2e-07
  exact at init (spread=0)                max|MoL - plain| = 0.000e+00
  router gets gradient under hard top-1   |grad| 4.2e+01, ALL E experts updated
  ceiling escape, experts diverged        key-set rank 18 with r=16
Value-side absorption regroups rather than factors:
    sum_j a_j up_v^(e_j) c_j = sum_e up_v^(e) ( sum_{j:e_j=e} a_j c_j )
E latent accumulators and E up-projections per STEP, nothing per key.

### 34.3 THE UNION IS CAPPED AT min(E*r, d_out), NOT E*r

I claimed E*r. The complement of span(up_k) has only d_out - r directions, so at the
real shapes (d_out = kv_heads*head_dim = 1024, r ~ 512) the cap is 1024 = 2r.
Measured: union dim 512 at spread=0, 1024 at spread=0.3, for both E=2 and E=4. So
E=2 already saturates the SPANNING gain; what E>2 buys is the per-token conditional
choice of WHICH r-dim slice of the 1024-dim key space to use -- the piecewise-linear
benefit, which does not saturate. mol2-spread vs mol4-spread separates the two.

### 34.4 Diverse init, and why exactness is the wrong target here

Identical-copy init is exact but leaves the router with ZERO output-based gradient at
step 0, since all experts compute the same thing. MoELoRA 2402.12851 documents this
regime: "the gating network shows no preference for any specific expert, resulting in
a routing process that appears random", and "the content learned by all experts
actually does not differ significantly". Router dynamics under literal identical-copy
init are an UNSTUDIED GAP -- no paper found addressing it.
--mol-spread gives each expert distinct directions from the complement of span(up_k).
Exactness is given up deliberately: MLA is ALREADY a lossy rank-r truncation, so the
comparison arm is inexact too, and a union of distinct rank-r decoders is a better
starting point than one global rank-r fit. What it CANNOT do is let a single token see
the full-rank map -- a rank-r decoder on c_j cannot reproduce W_K x_j for arbitrary
x_j. The union spans more; each token gets one slice, chosen for it.
BUG FOUND IN MY OWN INIT: the block offset (e*r) % (cols - r) WRAPPED, handing experts
0 and 2 the identical subspace (principal-angle cos 1.0) -- E/2 distinct decoders for
E experts. Replaced with a per-expert seeded random draw inside the complement;
measured cos 0.87-0.92 pairwise, no collisions.
The principled version -- per-cluster least-squares fits against the discarded
singular directions of the real W_K -- needs a calibration pass and is not built.

### 34.5 Hyperparameters, from the literature

  balance 0.01   Switch Transformer 2101.03961, alpha*E*sum_e f_e*P_e, published value
  z-loss 0.001   ST-MoE 2202.08906, NOT implemented yet
  noisy top-k    Shazeer 1701.06538, w=0.1; exposed as --mol-noise, default 0
  E             no saturation study exists for decoder routing; HydraLoRA reports
                k=2-4 "not a sensitive parameter"
  g/g.detach()  a straight-through variant (cf. Bengio 2013; van den Oord 2017
                VQ-VAE). No paper names the MULTIPLICATIVE form. Switch multiplies by
                g_e undetached, which would scale output by ~1/E and destroy
                exactness at init -- they train from scratch so they do not care.
CAUTION: MoH needed ~400B tokens and a two-stage schedule to stabilise converting a
pretrained model to routed heads. Our budget is 150 steps. Expect the router to be
the limiting factor, not the mechanism.

QUEUED as queue_v8: c0-mol4-spread, c0-mol4-copy, c0-mol2-spread.
`experts` deliberately sit in the DENSE 3e-5 group -- unlike every other new tensor
here they start at real weight magnitude; only `.router.weight` goes in the 6e-4 group.

## 35. MoL correction: a shared encoder CANNOT be saved by diverse decoders (2026-09-27)

Asked whether MoL's experts could be initialised as different subspaces so the
mixture "matches the full weights essentially". MEASURED, and the answer is no --
not with a shared encoder. Relative key-reconstruction error against the true
W_K x, d_c=48, k_out=128, 512 tokens:

    spread   rel error   vs plain MLA
      0.00     0.5772        --
      0.02     0.5774      +0.0%
      0.05     0.5785      +0.2%
      0.10     0.5827      +1.0%
      0.30     0.6259      +8.5%
      0.60     0.7542     +30.7%

THE REASON IS STRUCTURAL, not a poor choice of directions. With a shared `down`,
c_j carries only the top-r RIGHT-singular coordinates of W_K. The discarded energy
lives in V_rest^T x, which the encoder never computed, so NO up_k can reconstruct
it; and given that fixed c, the least-squares-optimal up_k is UNIQUE. Any deviation
from it strictly increases error. So my earlier "branch the decoder, share the
encoder" was right for the RANK-CEILING argument (#34.1) and wrong for the
RECONSTRUCTION argument.

### 35.1 The fix: branch BOTH, and the cache still costs only the index

    c_j = down_{e_j}(x_j)        r-dimensional, cached
    K_j = up_{e_j}(c_j)
Each token only ever needs ITS OWN expert's coordinates, so the cache stays
r + index -- the cost story is unchanged. And now each (down_e, up_e) pair can be
the best rank-r approximation of W_K RESTRICTED TO THE REGION OF INPUT SPACE ROUTED
TO e: a genuine piecewise low-rank fit, strictly better than one global rank-r fit,
because each region gets its own singular basis rather than sharing one.
Absorption is untouched: q^T K_j = (up_{e_j}^T q)^T c_j.

Init for that version, in increasing order of cost:
  (a) SVD-block split -- expert e gets (Sigma_e V_e^T, U_e) for the e-th rank-r
      block of W_K's SVD. Union spans full rank, no calibration needed. Expected
      gain is modest, since block 0 holds the largest singular values and will be
      best for most tokens; it helps atypical ones.
  (b) PER-CLUSTER weighted SVD -- k-means the calibration activations, then a
      covariance-weighted SVD of W_K per cluster. This is CARE generalised from one
      global whitened SVD to E local ones, and is the principled version. Needs a
      calibration pass collecting activations, which care.py's collect_covariances
      already has the machinery for (it currently keeps only the covariance).
NOT YET IMPLEMENTED. The current mol.py branches the decoder only.

### 35.2 Consequences applied

--mol-spread default 0.3 -> 0.0, since the measurement shows it only adds error.
queue_v8 arms changed from {mol4-spread 0.3, mol4-copy, mol2-spread 0.3} to
{mol4-copy, mol4-break 0.05, mol2-copy}: a small spread survives ONLY as a router
symmetry-breaker against the MoELoRA cold-start (#34.4), and 0.05 costs +0.2%.
NOTE queue_v8 had to be relaunched: it was already running and editing a live bash
script is unsafe, because bash reads by byte offset and the edit changed the length.

### 35.3 Local linear attention: the open choice, stated properly

Softmax attention is Nadaraya-Watson -- a locally weighted CONSTANT fit. Local
linear regression fits a sloped line through the neighbourhood and evaluates it AT
THE QUERY'S OWN LOCATION, which is lower-bias (Fan 1993: NW carries O(h) boundary
bias and is not minimax efficient; local linear is design-adaptive with O(h^2) bias
everywhere). It still sums to one -- a linear smoother reproducing constants AND
linear functions -- but THE WEIGHTS BECOME SIGNED, so the output leaves the convex
hull of the values. That is where the bias reduction comes from, and it is also the
bf16 stability risk.

Full version is O(T d^2): it needs S_2 = sum_j w_j delta_j delta_j^T, a d x d matrix
per query, ~256x attention at head_dim 256. NOT viable.

The 1-D version along the score axis IS viable AND ABSORBABLE. With s_j = q.k_j/sqrt(d)
(the logit already computed), fit v ~ beta_0 + beta_1 (s_j - s*). The only new
accumulator is nu = sum_j w_j s_j v_j, O(d_v) per key -- the same order as the value
accumulation already there -- and crucially
    nu = sum_j w_j s_j up_v c_j = up_v ( sum_j w_j s_j c_j )
so it is a SECOND LATENT ACCUMULATOR: decode absorption survives, zero extra cache.
Then beta_1 = (nu - m1 mu_v)/(m2 - m1^2 + eps), output mu_v + lambda beta_1 (s* - m1),
with eps a ridge (the variance of s collapses when all keys score alike) and lambda
a zero-init gate so the install is exact.

THE OPEN CHOICE is s*, the query's own coordinate on the s axis -- the query is not a
key, so it has none. Candidates: q.q/sqrt(d) (treat the query as a key, "distance
zero", the default I will use); max_j s_j; or a learned per-head constant. It sets
where on the fitted line the value is read off, hence the size and sign of the
correction. With the gate it is learnable either way.

## 36. Local linear attention: correct theory, wrong regime. NOT BUILT. (2026-09-27)

Proposal: replace softmax attention (which IS Nadaraya-Watson, a locally weighted
CONSTANT fit) with local LINEAR kernel regression, which still sums to one and is
"strictly an improvement over plain Nadaraya-Watson". The theory is right and the
idea does not survive our dimensions.

### 36.1 First, the formulation fixed -- there is no free evaluation point

I proposed a scalar covariate s_j = q.k_j/sqrt(d) and then had to invent an
evaluation point s*, asking which of q.q/sqrt(d), max_j s_j or a learned constant to
use. That question was an artifact of a bad reduction. THE QUERY ALREADY LIVES IN KEY
SPACE -- that is what makes q.k meaningful -- so the covariate is k_j, the evaluation
point is q, and the displacement delta_j = k_j - q is ZERO at the query BY
CONSTRUCTION. No choice exists. In the 1-D reduction the right covariate is the
signed displacement along the query direction,
    t_j = qhat.(k_j - q) = (q.k_j - |q|^2)/|q|
whose origin is t = 0. The |q|^2 I was calling "s*" is just the origin of the
displacement coordinate; it falls out of "evaluate at q".

Estimator: m_t = sum w_j t_j, beta_1 = (sum w_j t_j v_j - m_t mu_v)/(var_t + eps),
output = mu_v - beta_1 m_t. m_t is the weighted-mean OFFSET of the attended keys from
the query, so the correction removes the skew a lopsided neighbourhood induces --
exactly the boundary-bias correction local linear is known for.

### 36.2 VERIFIED: it is a proper linear smoother

Constant field 3.7 -> NW 3.700000, local-linear 3.699999. Reproduces constants, so
the effective weights sum to one, as claimed. They are also SIGNED, so the output
leaves the convex hull of the values -- where the bias reduction comes from, and a
bf16 stability risk.

### 36.3 MEASURED: bias reduction collapses with dimension and goes NEGATIVE at ours

Values exactly LINEAR in the keys (the most favourable possible case), T=400,
m = number of displacement directions fitted:

    d_head      m=1      m=2      m=4      m=8
        16     5.2%    54.7%    67.3%    76.2%
        64     1.3%    34.8%    39.7%    50.3%
       256    -0.6%   -19.8%   -10.8%    +2.7%      <- our head_dim

At head_dim 256 it HURTS. Mechanism, measured: local linear must estimate d+1
parameters from the EFFECTIVE sample size the kernel provides, ESS = 1/sum(w^2):

    d_head      T      ESS   d+1    ESS/(d+1)
       256    400    212.8   257       0.83
       256   4000   2110.0   257       8.21
        16    400    221.7    17      13.04

At d=256, T=400 there are FEWER effective points than parameters -- the fit is
under-determined and its variance swamps the bias it removes. This is the classical
curse of dimensionality for local polynomials (Fan & Gijbels): the O(h^2)
design-adaptive advantage is asymptotic in sample size at FIXED low dimension.
And the synthetic above is GENEROUS -- its attention is diffuse (ESS ~213 of 400),
while real attention concentrates far more sharply, lowering ESS further.

### 36.4 Decision

NOT IMPLEMENTED. The only positive cell is m=8 at T=4000 (+2.7%), which costs 8x the
per-key value accumulation for a gain inside the noise. Revisit only if head_dim
drops sharply or attention is deliberately made diffuse.

Worth keeping from the analysis: the 1-D version WAS absorbable --
sum_j w_j t_j v_j = up_v( sum_j w_j t_j c_j ), one extra latent accumulator, zero
extra cache. So the mechanism was shippable; it simply does not help.

## 37. RETRACTION of #36: the local-linear negative was an artifact of synthetic keys (2026-09-27)

#36 concluded local linear attention "hurts at our head_dim". That conclusion is
WITHDRAWN. Three errors, all pointed out by the user, all confirmed.

### 37.1 The test used isotropic random keys, which is the worst case

I generated K = q + isotropic Gaussian and V = A K + b with random A. Isotropic keys
fill all d directions, so the effective covariate dimension IS d and my ESS/(d+1)
argument follows trivially. Re-measured with the REAL W_Q/W_K/W_V from
ckpt/qwen3.5-4b-stageAB and the REAL per-layer input covariance from
cache/kv_covs_4b_mix.pt, sampling x ~ N(0, Sigma_real), participation ratio of the
key cloud's eigenvalues:

    layer   key eff. dim    ESS    ESS/eff-dim
        3          5.2     14.2        2.7
        7         13.5    273.9       20.3
       19         11.0    169.3       15.4
       31          7.0      5.8        0.8

EFFECTIVE DIMENSION IS 5-13, NOT 256 -- lower even than the MLA group ranks
(106-789), because the input anisotropy concentrates the keys. The synthetic test
inflated it 20-50x. The correct denominator for "can a local slope be fitted" is
5-13, so ESS/eff-dim is 3-20 for most layers: a FAVOURABLE regime, the opposite of
what #36 claimed.

### 37.2 q and k do NOT live in the same space

q = W_Q x and k = W_K x are different linear images of the residual stream. They
share a DIMENSION, not a space: the score x_t^T W_Q^T W_K x_j is a non-symmetric
bilinear form, not an inner product on a common space. So delta_j = k_j - q (the
whole basis of #36.1's formulation, and of the "s*" question before it) subtracts
incommensurable quantities -- q's DIRECTION is meaningful as a functional on key
space, its MAGNITUDE has no relation to key norms. Both formulations were wrong.

### 37.3 The real open question, stated correctly

NW IS ALREADY UNBIASED AT THE WEIGHTED-MEAN KEY. For v linear in k,
    sum_j w_j v_j = A (sum_j w_j k_j) + b = A kbar + b
so at kbar there is NOTHING to correct -- measured, both estimators agree to
floating point (0.0% at layers 3/7/19/31). The bias local linear removes is exactly
the offset between kbar and the location where the value is actually wanted. So the
mechanism's entire value rests on defining an EVALUATION POINT in key space, and
that is the thing #37.2 shows we cannot get by treating q as a key.

Candidate resolution, NOT yet confirmed as what was intended: step from kbar along a
key-geometry-weighted direction, e.g. Sigma_K q, using the query only as a functional
and the key covariance to fix the scale. That never subtracts q from a key, so it
respects W_Q != W_K, and is built from keys. It introduces a step size, which is a
real free parameter rather than a derived one.

### 37.4 Process note

I reported a synthetic negative as though it settled a design question, after
explicitly telling the user I would measure before building. Measuring the wrong
thing is not measuring. The realistic test needed no GPU and no new data -- the real
weights and the real covariance were already on disk.

## 38. Grouping costs nothing at matched rank; local-linear validated but modest (2026-09-27)

### 38.1 ungrouped150: a clean null

GSM8K 1078/1319 = 81.7% against base150's 1070/1319 = 81.1%, both --dial nope, 150
steps, IDENTICAL per-layer rank, only the partition differing.
PAIRED McNemar: 117 gains, 109 losses, +0.61 points, z = +0.53, p = 0.64.

The hypothesis behind _mk_ungrouped_groups.py -- that grouping pays a PERMANENT
expressiveness cost for an init-time benefit adaptation erases -- is refuted on the
cost side. There is no detectable cost, and no benefit either. Consistent with the
earlier off-block measurement (cross-group weights at 0.3-0.5% of on-block, using 5%
of their reachable displacement): the restriction was never binding because training
never wanted to cross it.
Mild practical argument for dropping grouping in line 1 -- same accuracy, same 334
tok/s, one fewer moving part, matches DeepSeek/TransMLA, and MoL assumes a shared
encoder -- but it is a coin flip, so not worth changing for its own sake.
c0-ungrouped150 (queue_v7) will say whether it matters with position intact.

### 38.2 Local linear attention: the direct mapping, and what it is actually worth

Mapping, with no tricks: covariates X_i = k_j, responses Y_i = v_j, kernel weight
K_h(X_i - x) = w_j = softmax(q.k_j/sqrt(d)), evaluation point x = q, output = beta_0
of argmin sum_j w_j ||v_j - beta_0 - B(k_j - q)||^2. The Q/K scale mismatch does NOT
break this: B is SOLVED FOR, so it absorbs any mismatch. My earlier "s*" invention
and 1-D reduction were both unnecessary.

WHY THE NAIVE FORM FAILS, measured: 26-56% of the (kbar - q) displacement lies
OUTSIDE the key cloud's 99%-energy span, so B is undetermined in exactly the
direction the extrapolation needs. q is only 1.2-1.8 Mahalanobis units from the key
cloud (a typical key is 1.0), so it is not implausible -- the problem is the
DIRECTION, not the distance. This is the concrete consequence of W_Q != W_K.

FIX, which is what "use keys instead" means: evaluate at the KEY-SPAN PROJECTION of
the query, qp = kbar + P_span(q - kbar). Kernel weights still use the true q, so the
attention pattern is untouched; only value aggregation changes.

    with v EXACTLY linear in k (no irreducible residual):  62-95% error reduction
    m = 4 fitted directions already gives 56-73%; saturates by m = 8-16

BUT WITH REAL V (v = W_V x, not a function of k at all) the gain mostly evaporates:
    layer   NW err   LL best    gain    key eff.dim
        3    10.36    12.92   -24.7%        5.1
        7    10.07     9.49    +5.7%       13.8
       11     9.06     8.58    +5.3%       18.0
       19    11.35    11.29    +0.5%       11.0
       27    25.71    19.71   +23.4%        6.1
       31    60.49    26.61   +56.0%        1.6
IT HELPS PRECISELY WHERE NADARAYA-WATSON IS WORST. Layer 31 has the most degenerate
key cloud (effective dimension 1.6) and by far the largest NW error, so its
neighbourhood is maximally lopsided and there is the most bias to remove. Layer 3 has
low NW error already, so fitting a slope adds variance and HURTS.

DESIGN CONCLUSION: per-head ZERO-INIT gate, m = 4 directions, ~4x the per-key value
accumulation, absorbable as 4 latent accumulators (sum_j w_j delta_ja v_j =
up_v(sum_j w_j delta_ja c_j)). Training turns it on where it pays and leaves lambda
at 0 elsewhere. Expect 2 of 8 layers to use it.

LIMITATION, stated: the held-out test centres the kernel at a real key k*, which is
not the attention situation; the q-projection test matched attention geometry but used
a synthetic linear target. Neither is the real thing -- only a training run is.

### 38.3 Key clouds are far lower-dimensional than the allocated rank

Participation ratio of the real key cloud (real W_K, real input covariance): 1.6 to
23.4, against allocated group ranks of 106-789 and 99%-energy ranks of 200-218 per
head. And the whitened joint [K;V] spectrum needs 7374 total for 99% energy against
4096 allocated -- so the 8094 plan is ~the 99% requirement, not an arbitrary doubling.
Shallow layers are the starved ones (layer 3: 359 allocated of 1282 needed; layer 7:
188 of 1105) while deep layers are matched (layer 31: 606 of 597). Same direction
MatryoshkaKV reports empirically. CAVEAT: water-filling optimises marginal error per
unit rank, not energy fraction, so "inverted" is not automatically "wrong".
All of this is under a Gaussian surrogate x ~ N(0, Sigma_real), not real activations.

## 39. Local linear must be LEARNED, not solved. Three measurements invalidated. (2026-09-27)

I implemented local-linear attention as a closed-form weighted least-squares solve,
(X^T W X)^-1 X^T W Y, per query, per forward pass. Wrong, for two reasons.

### 39.1 It makes the whole evaluation answer the wrong question

Bias, variance, ESS, effective dimension: all of that describes how well a FIXED
estimator recovers E[v|k] from given data. A trained network is not estimating
E[v|k]; it is learning a function that minimises loss, and W_Q, W_K, W_V are all
free to compensate for Nadaraya-Watson's bias already. So NONE of these bear on
whether a trained model gains:
  * the 5.2%/1.3%/-0.6% synthetic sweep (#36) -- already retracted in #37
  * the 62-95% key-span-projection result (#38.2)
  * the held-out real-V table, +56% on layer 31 down to -24.7% on layer 3 (#38.2)
All three measure a closed-form fit's estimation quality. The question was always
whether the FORM gives a trained model a useful inductive bias, which only a
training run answers.

WORSE: because VeRA adapts q_proj/k_proj/v_proj, the model can RESHAPE THE KEY
GEOMETRY to make the correction useful. Every offline number assumed fixed
projections, so they cannot even bound the trained outcome.

### 39.2 The solve created every implementation problem I complained about

The d x d inverse, the ridge, the ill-conditioning, the 26-56% of (kbar - q) lying
outside the key span, the "few hours of delicate work" -- all of it exists ONLY
because the slope was being solved for. Learn the coefficients and none of it does.
The evaluation-point question that cost three exchanges (s*, then q, then the
key-span projection) also dissolves: there is nothing to evaluate, only moments to
combine.

### 39.3 The form that ships

    z_{j,a} = u_a . k_j                 m learnable directions
    mu_v    = sum_j w_j v_j             ordinary attention output
    nu_a    = sum_j w_j z_{j,a} v_j
    m_a     = sum_j w_j z_{j,a}
    out     = mu_v + sum_a Lambda_a (nu_a - m_a mu_v)      Lambda ZERO-INIT

(nu_a - m_a mu_v) is the weighted covariance between displacement along direction a
and the values -- exactly the numerator local linear divides by S_2. Lambda learns
the scale instead of dividing. Per-head gates, so training raises Lambda only where
it pays (#38.2 predicts layers 27 and 31, not layer 3).

EVERYTHING TRAINS JOINTLY with VeRA, the latents and the norms. Deliberately NOT a
one-shot conversion followed by a frozen fit.

NO ATTENTION MATRIX: nu_a is attention with values replaced by z_{j,a} v_j, so all
m+1 aggregations are ONE fused call with a widened value tensor
[V, z_1 V, ..., z_m V, z_1, ..., z_m]. Value width d_v -> d_v(m+1) + m, 5x at m=4.

MEASURED at real dims (d_k 256, 16 heads, m=4), src/mercurius/surgery/moment_attn.py:
    fused widened call == explicit weighted sums    max err 6.0e-08
    exact softmax attention at install              0.0
    params                                          1,088/layer, 8,704 total
    KV CACHE                                        UNCHANGED, 0 extra bytes
    decode                                          m latent accumulators per STEP
    absorption  z = (up_k^T u_a).c,  nu = up_v(sum_j w_j z_{j,a} c_j)   holds to fp32

### 39.4 Remaining work and the risk to watch

Wiring. The correction belongs between the attention aggregation and
attn_output_gate/o_proj, since it corrects the AGGREGATION not the projection --
so it patches Qwen3.5's attention forward rather than swapping a module, unlike
every latent wrapper today. The 5x value width at seq 8192, 16 heads is ~336 MB per
layer per batch element against 67 MB now; peak is 11.3 GB with headroom, so it
should fit, and m is the dial if it does not.

### 39.5 Process

Three separate measurement rounds, each answering a question the design had already
moved past, and each reported as though it settled something. The user identified the
error each time: the synthetic data, then the Q/K geometry, then the closed-form fit
itself. The lesson is not "measure more" -- it is to check that the quantity measured
is the one the decision turns on.

## 40. CORRECTION to #39: attention already IS online regression (2026-09-27)

#39 concluded local linear "must be LEARNED, not solved" and replaced the closed-form
solve with learnable directions and per-head gates. That was backwards.

ATTENTION IS ALREADY ONLINE KERNEL REGRESSION. out = sum_j w_j v_j with
w_j = softmax(q.k_j/sqrt(d)) IS Nadaraya-Watson -- a kernel-weighted CONSTANT fit,
solved in closed form at every step from the cached K/V. The closed-form solve is not
a defect to parameterise away; it IS the operator. Switching to local linear means
switching the REGRESSION TYPE, online, from the same cache, with NO NEW PARAMETERS.
I turned a parameter-free estimator swap into a parameterised module. Reverted.

### 40.1 What "let it adapt" actually meant

Not new parameters -- the EXISTING ones. W_Q/W_K/W_V/W_O were trained against an NW
aggregator, so switching the operator leaves them mismatched, exactly as MLA, GDN-2
and NoPE leave them mismatched. The response is RECOVERY TRAINING. Which means the
operator is deliberately NOT exact at install, and a gate would be the wrong idea:
there is nothing to anneal, there is a surgery to recover from.

### 40.2 Which is why none of my offline tests could decide it

Every measurement (#36 retracted, #38.2's 62-95% and the held-out table, #39.1's
framing) evaluated the operator AT THE CURRENT WEIGHTS. But the current weights are
fitted to NW. Reconstruction error before adaptation predicts nothing -- the same
reason step-0 numbers do not predict post-recovery numbers anywhere else in this
project. WE HAVE NOT SEEN THE DOWNSTREAM EFFECT, and that is the only thing that
decides it: swap the operator, run recovery, measure GSM8K like every other arm.

### 40.3 The shipped form, verified

src/mercurius/surgery/moment_attn.py, LocalLinearAttn:
    beta = argmin_{b0,B} sum_j w_j || v_j - b0 - B (k_j - q) ||^2 ;  out = b0
projected onto m FIXED directions (a buffer precomputed from calibration, like the
MLA covariances), plus a ridge scalar.
    fused widened path == explicit per-query weighted least squares   6.3e-07
    learnable parameters                                             0
    value width dv=256, m=4                                          256 -> 1294 (5.1x)
    per-query solve                                                  5x5
    KV cache                                                         unchanged
Every quantity is a kernel-weighted sum, so ONE fused attention call with values
widened to [V, z_a V, z_a, z_a z_b] yields mu_v, nu_a, m_a and S_ab with no T x T
matrix. Absorption holds: z = (up_k^T u_a).c and nu_a = up_v(sum_j w_j z_{j,a} c_j).

### 40.4 Next

Wire it between the attention aggregation and attn_output_gate/o_proj, pick the
direction basis per layer from calibration (top-m of W_K Sigma W_K^T, which #38.3
already computes), then run it as a c0 arm WITH recovery and measure GSM8K against
ablate-rope150's 86.4%. Memory risk: the 5.1x value width at seq 8192.

### 40.5 Process, fourth time

#36 wrong (synthetic keys), #37 wrong (q as a key-space point), #39 wrong (learnable
instead of parameter-free). Each time I measured something and reported it as
settling a design question, and each time the error was in what I chose to measure,
not in the measurement. The recurring fault: evaluating a surgery at the PRE-RECOVERY
weights and treating that as evidence about the POST-RECOVERY model, which this
project has documented as invalid since #27.

## 41. MoL init, settled by measurement: disjoint is vacuous, per-cluster halves the error (2026-09-27)

### 41.1 Identical-copy init: hard top-1 does NOT differentiate experts

I assumed hard top-1 would separate identical experts, because each sees a disjoint
token subset. MEASURED, E=4, identical init: pairwise cosine between EXPERT GRADIENTS
is 0.889. Not ~0. A random 1/E subsample of the same distribution carries essentially
the full-data gradient, so the shared population term dominates and only ~11% differs.
--mol-spread changes nothing (0.891 vs 0.889), so `mol4-break` was dropped: it would
have measured an init difference that provably does not alter learning dynamics, at a
cost of +0.2% init error. `mol2-copy` dropped too -- if experts do not differentiate,
E=2 and E=4 are the same null.
A load-balance loss cannot fix this: it balances token COUNTS, not expert CONTENT.
MoELoRA's actual fix was a contrastive loss on expert OUTPUTS.

### 41.2 Disjoint subspaces: PROVABLY VACUOUS

Proposed: give each expert a disjoint block of W_K's SVD, so E=2 tiles the output
space exactly (r=512, d_out=1024 ungrouped, complement = 512 = exactly one more
block). Union = full rank. It sounds ideal and it does nothing.

Reconstruction of the true keys W_K x, real weights, real input covariance, ORACLE
routing so the router cannot be blamed, r=512, E=2:

    layer   plain MLA   disjoint   per-cluster
        3      0.2789     0.2789        0.1375
        7      0.2564     0.2564        0.1220
       19      0.1652     0.1652        0.0833
       31      0.1792     0.1792        0.0907

DISJOINT EQUALS PLAIN MLA TO FOUR DECIMALS. The singular spectrum is concentrated, so
block 0 is the better choice for EVERY token and the oracle never picks block 1.
Selection has nothing to select. Building it would produce an exact null by
construction.

### 41.3 What works: per-cluster, and the reason is the opposite of disjoint

Per-cluster whitened SVD -- k-means the calibration activations, then the top-r
directions of W_K in each cluster's own whitened geometry -- HALVES the error at the
same r + index cache. The mechanism is that every expert KEEPS the dominant
directions and they differ only in the TAIL, each fitted to its own input region.
DIVERSITY MUST BE IN THE TAIL. Diversity in the head throws away signal, which is
exactly what disjoint does.

Note the comparison this sets up: ~2x error reduction at r + index cache, against
doubling the rank to 8094, which costs 2x the cache for a comparable effect. If the
downstream numbers follow the reconstruction numbers, per-cluster MoL is strictly
better than buying rank. (Untested downstream -- and #40 is the standing warning that
pre-recovery reconstruction error does not settle post-recovery accuracy.)

### 41.4 Requires routing the ENCODER, per #35

With a shared `down`, c carries only the top-r RIGHT-singular coordinates, so an
expert whose subspace differs below that would multiply coordinates never computed.
Routed encoder keeps the cache at r + index because each token needs only its own
expert's coordinates. Implementation must sit inside transmla.py's latent
construction: the SVD is computed there and the discarded block is thrown away, and
k_ref/v_ref only survive under blend=True, so the factors cannot be recovered
afterwards. NOT YET BUILT.

### 41.5 queue_v9 as launched: 5 arms, ungrouped, c0, 150 steps

    ungrouped150   the CONTROL (grouped ablate-rope150 is the wrong reference)
    taps150        --mla-taps 1
    conv150        --mla-conv 4 --mla-conv-where both
    gate150        --mla-gate xatlu
    f2a2-150       --f2a2
mla8094 dropped (#40-adjacent): as a grouped plan it was degenerate -- single-head
groups cap at min(2560, 512) and layers 3/15/31 sat at 98.8/97.3/100.0% of full rank,
layer 31 LITERALLY UNCOMPRESSED -- and even ungrouped it is only 51% compression
against 75% for 4096.

## 42. Per-cluster is INIT ONLY, so #41's numbers may wash out (2026-09-27)

Clarification from the user, and it changes what #41.3's measurements mean: per-cluster
whitened SVD is an INITIALISATION. The router is an ordinary learned MoE router, the
expert weights train, and nothing pins either to the k-means partition. So the 2x
reconstruction improvement is an INIT property, not a prediction.

### 42.1 This project has evidence both ways

  * #25: water-filling's ALLOCATION advantage was 92% recovered by training -- worth
    0.062 nats and 0.5 sigma at the end. Init washed out.
  * #29.3: Palu's worse init stayed 7 POINTS behind CARE at 3B tokens. Init did not
    wash out. And that comparison -- plain SVD vs covariance-aware whitening -- is
    much closer in kind to per-cluster-vs-global than #25's allocation shuffle was.
So "init washes off" is not a law here; it held for allocation and failed for
factorisation quality. Per-cluster is the second kind.

### 42.2 The durable justification is SYMMETRY BREAKING, not starting error

#41.1 measured expert gradients at 0.889 cosine under identical-copy init, so the
mechanism may never differentiate in 150 steps. A differentiated init is therefore not
buying a head start on reconstruction error -- it is what makes MoL TRAINABLE AT ALL.
That reason cannot wash out: if the experts never separate there is nothing to wash.
This is a better argument for per-cluster than the 2x, and it survives #42.1 either way.

### 42.3 What the E/r sweep is legitimately for

CACHE IS STRUCTURAL: r + index stays r + index whatever training does to the weights.
ACCURACY AT A GIVEN CACHE is the part that may wash. So the sweep's only sound use is
to find (E, r) pairs whose INIT error at HALF the cache already matches plain MLA at
full cache -- e.g. E=4 at r=256 (87.5% compression) against plain at r=512 (75%).
Those are candidates to test downstream; the sweep cannot itself decide them.
Choosing E on init error alone would repeat the #40 mistake.

## 43. E/r sweep for per-cluster MoL: E=4 >> E=2, and E=8 at r=256 halves the cache (2026-09-27)

Key reconstruction error vs the true W_K x. Real weights, real per-layer input
covariance, REAL k-means routing (not an oracle). Cache/token/layer = r + index; E
does NOT enter the cache.

    config              cache  compress      L3      L7     L19     L31
    plain   E=1 r=512     512     75.0%  0.1633  0.1436  0.0972  0.1064
    cluster E=2 r=512     512     75.0%  0.1375  0.1220  0.0833  0.0907
    cluster E=4 r=512     512     75.0%  0.0958  0.0744  0.0516  0.0597
    plain   E=1 r=256     256     87.5%  0.3062  0.2945  0.2249  0.2322
    cluster E=2 r=256     256     87.5%  0.2844  0.2740  0.2100  0.2166
    cluster E=4 r=256     256     87.5%  0.2442  0.2332  0.1794  0.1859
    cluster E=8 r=256     256     87.5%  0.1541  0.1459  0.1129  0.1187
    cluster E=4 r=128     128     93.8%  0.3812  0.3852  0.3232  0.3217
    cluster E=8 r=128     128     93.8%  0.3272  0.3298  0.2769  0.2769

### 43.1 E=4 is much better than E=2, and does not saturate at 4

At fixed r=512: E=2 buys -16%, E=4 buys -41% (layer 3; same ordering on 7/19/31).
I would have tested only E=2 without being told to include E=4.

THIS ALSO CORRECTS #34.3's objection to E>2. That cap -- union <= min(E*r, d_out) =
2r at r=512 -- is about SPANNING. The gain here is not spanning, it is per-region
specialisation, so it keeps paying past E=2, exactly as #41.3's tail-diversity
argument implies. The spanning cap was the wrong lens for choosing E.

### 43.2 E=8 at r=256 matches plain r=512 at HALF the cache

vs plain E=1 r=512, which has TWICE the cache:
    cluster E=2 r=256    +74.2%  +90.9% +116.1% +103.6%   worse
    cluster E=4 r=256    +49.6%  +62.4%  +84.6%  +74.7%   worse
    cluster E=8 r=256     -5.6%   +1.6%  +16.2%  +11.6%   PARITY
    cluster E=4 r=128   +133.5% +168.2% +232.6% +202.4%   cliff
So 87.5% compression at roughly the quality of 75%. r=128 is a cliff in every
configuration, so 256 is the floor.

### 43.3 Costs and caveats

PARAMETERS: E=8 at r=256 is ~75 M expert params (8 experts x 1.18 M x 8 layers)
against ~19 M for plain r=512 -- 4x the WEIGHTS to halve the CACHE. Good trade for
inference memory, but it grows the model.
CAVEATS, per #42: these are INIT quality with k-means routing; a learned router may go
anywhere, and #25 measured an allocation advantage washing out 92%. Also K only (not
V), and a Gaussian surrogate x ~ N(0, Sigma_real) rather than real activations.

### 43.4 Two candidates to build and test downstream

    E=4 at r=512    same cache as now (75%), best quality
    E=8 at r=256    half the cache (87.5%), parity with plain at 75%
Both need the routed ENCODER (#41.4) inside transmla.py's latent construction, plus a
calibration pass for the k-means. Neither is decided by the table above -- only by
GSM8K after recovery, against c0-ungrouped150.

## 44. #35 WAS WRONG: latent-routed decoders work, and need no cached index (2026-09-27)

#35 concluded that a SHARED encoder "provably cannot be saved by diverse decoders",
because given a fixed c the least-squares-optimal up_k is unique. That is true of a
single GLOBAL fit and false of a PIECEWISE one -- a different up_k per latent region
is strictly better, for exactly the reason per-cluster beats global everywhere else in
these notes. What #35 actually measured was that RANDOM diversity hurts (+8.5% at
spread 0.3); I generalised that to all decoder-only routing, which does not follow.

### 44.1 The design, which removes the index entirely

Route on the LATENT, not on x:
    c_j = down(x_j)              SHARED encoder, cached, r dims
    e_j = router(c_j)            a deterministic function of what is already cached
    K_j = up_k^(e_j) c_j
Because e_j is recomputable from c_j, NOTHING EXTRA IS STORED: the cache is
byte-identical to plain MLA, so no change to cache format, quantisation or eviction.
(Routing on x instead would need the index, since x is not cached. And if the ENCODER
were routed, c_j would depend on e_j and routing from c_j would be circular -- so the
two designs are alternatives, not composable.)

COST is not quite zero: reading a stored index is free, while recomputing router(c_j)
is E*r MACs per cached key per step -- at r=512, E=4 about 4x the score computation.
So the index is best understood as a 2-bit CACHE for a recomputable value. Storing it
remains the better trade; the point is that it is now OPTIONAL.

### 44.2 Measured: 13% at E=4, 30-35% at E=8, at zero cache cost

Key reconstruction vs true W_K x, real weights, real per-layer covariance, shared
encoder r=384, k-means in latent space, ridged per-region least squares:

    layer   global      E=2      E=4      E=8    E=4 gain   E=8 gain
        3   0.2413   0.2311   0.2092   0.1697      13.3%      29.7%
        7   0.2200   0.2106   0.1904   0.1547      13.4%      29.7%
       11   0.1837   0.1759   0.1591   0.1187      13.4%      35.4%
       19   0.1547   0.1481   0.1341   0.0999      13.3%      35.4%
       27   0.1018   0.0975   0.0882   0.0658      13.4%      35.3%
       31   0.1673   0.1602   0.1449   0.1177      13.4%      29.7%
Remarkably uniform across layers, which is itself evidence the effect is structural
rather than a quirk of one layer's spectrum.

### 44.3 Against the routed-encoder version

    shared encoder + latent routing, E=4    cache = r            -13.4%
    shared encoder + latent routing, E=8    cache = r            -30 to -35%
    routed encoder + x routing, E=4 r=512   cache = r + index    -41%  (#43)
The routed encoder is stronger; the shared-encoder variant is free and needs no
plumbing change. Both worth building; the shared one is strictly simpler and should
probably be tried first for that reason.

### 44.4 Caveats, unchanged from #42

Init-time only: the router is learned and may abandon the k-means partition. Gaussian
surrogate x ~ N(0, Sigma_real), not real activations. K only, not V. And #25 measured
an allocation advantage washing out 92%, so none of this predicts post-recovery GSM8K.

### 44.5 A tooling note

The first two attempts at this measurement CORE-DUMPED. Cause was torch.linalg.lstsq
and cholesky+solve_triangular on CPU under load, not memory (44 GiB free, eigh fine).
Replacing them with eigh-based whitening and explicitly ridged normal equations
    G = C^T C + lam*mean(diag(G))*I ;  up = solve(G, C^T K)
is robust and should be the default for these offline fits.

## 45. The rank x RoPE 2x2 completed: capacity headroom is <=3 points (2026-09-27)

step0-rope8094: 1178/1319 = 89.3%. STEP 0, NO TRAINING in any cell.

                     r=4096 (75% compress)   r=8094 (51%)
    RoPE (dial c0)          86.4%                89.3%
    NoPE                    75.4%                83.3%
    unmodified base         90.5%

PAIRED, n=1319:
    rank 4096->8094 UNDER RoPE     96 gain,  57 loss   +2.96p  z=+3.15  p=0.002
    rank 4096->8094 under NoPE    168 gain,  63 loss   +7.96p  z=+6.91  p=3.3e-12
    NoPE->RoPE at r=4096          198 gain,  53 loss  +10.99p  z=+9.15  p=7.2e-21
    NoPE->RoPE at r=8094          125 gain,  46 loss   +5.99p  z=+6.04  p=1.2e-09

### 45.1 Rank is worth 2.7x more without RoPE

+2.96 under RoPE against +7.96 under NoPE. #28.4 inferred "at least 1.9x" from three
cells; the fourth puts it at 2.7x. The effects are strongly sub-additive, as that
entry predicted (10.99 + 7.96 would overshoot the base).

### 45.2 The surgery is nearly lossless at step 0 once position is intact

GDN-2 lift + MLA at 51% compression + native partial RoPE = 89.3%, only 1.2 points
below the unmodified 90.5%, WITH NO TRAINING AT ALL. So the damage fought all day was
overwhelmingly NoPE (#32.2's decision), not the conversion. It also means the
GDN-2 + MLA line starts from a much better place than any number measured before this.

### 45.3 THE CEILING FOR EVERY CAPACITY MECHANISM IS <=3 POINTS

At 75% compression with RoPE we sit at 86.4%. Buying capacity all the way down to 51%
compression gains 2.96 points; reaching the unmodified base would be 4.1. So MoL,
taps and conv are collectively chasing AT MOST ~3 points, and must do it WITHOUT the
2x cache that r=8094 costs.

CONSEQUENCE FOR MoL: its measured claim is PARITY AT HALF THE CACHE (E=8, r=256 ~=
plain r=512, #43.2), which moves along the COMPRESSION axis, not the quality axis.
Given only 3 points of quality headroom, compression is the more promising axis and
MoL should be evaluated as "same accuracy, 87.5% compression instead of 75%".

RESOLUTION PROBLEM, stated plainly: paired sigma on a difference is ~1.4 points at
n=1319, so a mechanism worth 1-2 points sits at the edge of detectability. Any arm in
queue_v9 that lands inside +-3 points of the control is UNRESOLVED, not negative, and
saying otherwise would repeat the HumanEval error of #26. Distinguishing small real
effects would need a larger eval set or paired seeds, not a louder claim.

## 46. Control established at 87.2%; recovery again shows nothing under c0 (2026-09-27)

c0-ungrouped150: GSM8K 1150/1319 = 87.2%, ppl@8192 9.093. Best trained arm of the day
on both metrics. THIS IS THE CONTROL for every mechanism arm in queue_v9.

PAIRED, n=1319:
    grouped -> ungrouped, under c0, trained    76 gain, 66 loss  +0.76p  z=+0.84  p=0.45
    step0 grouped -> trained ungrouped         84 gain, 73 loss  +0.83p  z=+0.88  p=0.42
    control -> capacity ceiling (2x cache)     88 gain, 60 loss  +2.12p  z=+2.30  p=0.026

### 46.1 Dropping grouping is confirmed free

+0.76, p=0.45 -- a second null after #38.1's +0.61 (p=0.64) under nope. The
simplification taken on simplicity grounds costs nothing, which is the best available
outcome for such a change. Perplexity actually improved: 9.093 against rope150's
9.172, though #27 is the standing warning that ppl and GSM8K disagree here.

### 46.2 RECOVERY SHOWS NOTHING UNDER c0, for the second time

Training 150 steps from the step-0 grouped model to the trained ungrouped one gains
+0.83 points at p=0.42 -- INDISTINGUISHABLE FROM ZERO. Combined with #27's finding
that step0-rope150 and rope150 were both exactly 86.4%, recovery has now twice failed
to register under c0. The architecture is carrying essentially all of the quality.
(Strictly this contrast also changes the grouping, but 46.1 measures that as null, so
the training is the only other difference.)

### 46.3 The resolution problem is now binding

Control 87.2% against the capacity ceiling 89.3% (step0-rope8094, 2x cache) leaves
2.12 points for taps/conv/gate/f2a2 to compete over, and the paired sigma on a
difference is ~1.4 points. A mechanism must capture MORE THAN HALF the entire
remaining capacity gap to reach significance. So all four arms tonight are expected to
land UNRESOLVED, and that is a statement about the experiment's power, not about the
mechanisms. Reading a 1-point gap as a result would repeat #26.

TO DISTINGUISH 1-2 POINT EFFECTS we would need a larger eval set or paired seeds per
arm. n=1319 cannot do it.

### 46.4 Where the leverage actually is

Recovery: twice measured as null under c0. Quality via capacity: 2.1 points, below
resolution. COMPRESSION: MoL's measured claim is parity at 87.5% instead of 75%
(#43.2, #44.2), which is a STRUCTURAL saving -- it does not require detecting a small
accuracy difference at all, only showing that accuracy does NOT fall when the cache is
halved. That is a test this eval can actually run, because the null hypothesis is the
one we want.

## 47. Recovery DAMAGES reasoning under RoPE; the truncation fix masks it (2026-09-27)

Asked directly whether NoPE or MLA was recovered. Re-derived, and the answer to both
is NO. Split of the step0 -> 150-step gain by whether the STEP-0 generation hit the
768-token cap:

    arm                 total      p     from TRUNCATED   from COMPLETED
    NoPE + MLA 4096    +5.76p  3.4e-06        +62 items    +14 items  z=+1.01 p=0.35
    NoPE + MLA 8094    +2.58p    0.012        +32 items     +2 items  z=+0.17 p=0.93
    RoPE + MLA 4096    +0.08p        1        +35 items    -34 items  z=-3.10 p=0.002

COMPLETED-only is the reasoning measure: the model finished its answer in both arms,
so nothing there is termination.

### 47.1 The MLA arm's significant gain was 94% truncation

mla8094's +2.58p at p=0.012 is real as a total, which is why it reads as "MLA
recovered". But +32 of the +34 items are truncation repair and the reasoning component
is +2 items out of 1319 at p=0.93. Separately: what made mla8094 SCORE better in
absolute terms (85.9% vs base150's 81.1%) was TWICE THE RANK AT CONSTRUCTION, not
training. More rank recovered quality; recovery training did not.

### 47.2 NEW, and under-reported in #27: recovery is NOT neutral on reasoning

Under RoPE it makes completed-item accuracy SIGNIFICANTLY WORSE -- -34 items,
z=-3.10, p=0.002 -- and nets to +0.08 only because it repairs 35 truncations in the
same run. #27 recorded the -34 but did not flag that it is significant. So 150 steps
of our recipe actively degrades reasoning on items the model could already finish, and
the termination fix hides it.

### 47.3 What this implies

The surgery is nearly lossless once position is intact: 89.3% at step 0 against a
90.5% base (#45.2). The RECOVERY RUN is the part doing damage. So the recipe is
mis-tuned or too short to help while being long enough to hurt -- which is a stronger
argument for the data-and-steps work (#32.4) than any architecture arm, and it means
tonight's mechanism arms are being measured on top of a training run that is itself
net-harmful to reasoning.

CAVEAT: the truncation split is not a perfect causal decomposition -- a model that
stops rambling may also reason differently. "Completed in both arms" is the cleanest
available reasoning subset, not a proof.

## 48. RETRACTION of #47 and of #27's decomposition: my split was biased (2026-09-27)

#27 and #47 both split the step0 -> 150-step gain by whether THE STEP-0 GENERATION was
truncated, then reported the two halves. That conditioning is a SELECTION EFFECT and it
biases both halves in opposite directions:
  * the step-0-truncated group is where step-0 did badly, so any second model scores
    better there through REGRESSION TO THE MEAN alone;
  * the step-0-completed group is where step-0 did well, so any second model scores
    worse there for the same reason.
I built the bias in and then reported the downward half as a finding ("recovery
significantly degrades reasoning, -34 items, p=0.002"). WITHDRAWN.

The user spotted it by asking why the two halves pointed in incoherent directions.

### 48.1 The unbiased analysis: two symmetric statistics

Truncation counts are MARGINAL (no conditioning). Accuracy is measured only on items
BOTH arms finished, which conditions on both models identically.

    arm             trunc@0  trunc@150 | both-done   acc@0   acc@150   delta      p
    NoPE + r4096        138         53 |     1158    83.8%    85.9%   +2.16p   0.074
    NoPE + r8094         82         48 |     1210    88.8%    90.2%   +1.40p   0.15
    RoPE + r4096         76         45 |     1215    91.5%    90.1%   -1.40p   0.11

### 48.2 What survives, and what does not

SURVIVES: the termination fix is real and large (138->53, 82->48, 76->45), and it is a
clean marginal statistic.
DOES NOT SURVIVE: "recovery buys termination, not reasoning" (#27) and "recovery
damages reasoning" (#47). On the symmetric subset the reasoning effects are +2.16
(p=0.074), +1.40 (p=0.15) and -1.40 (p=0.11) -- ALL UNRESOLVED, with the NoPE arm
leaning POSITIVE.
So the correct statement is: RECOVERY DEMONSTRABLY FIXES TERMINATION, AND ITS EFFECT ON
REASONING IS BELOW OUR RESOLUTION. That is the same power problem as #46.3, not
evidence of harm. #27's "82% of the gain is truncation" figure was computed the biased
way and must not be quoted.

### 48.3 Consequence for the project's conclusions

#32.4 justified prioritising data and steps partly on #27's decomposition. That
justification is weaker than stated -- the token-budget argument (#30, #43) stands on
its own, but "recovery does nothing for reasoning" was never established.
The headline totals themselves were never in doubt and remain correct: +5.76p
(p=3.4e-06), +2.58p (p=0.012), +0.08p (p=1.0) for the three arms. What was wrong was
the attribution of those totals between termination and reasoning.

### 48.4 The lesson, which is the same one as #37 and #40

Conditioning an A/B comparison on one arm's outcome is not a decomposition, it is a
biased estimator. When a split is needed, condition on something SYMMETRIC in the two
arms, or on a covariate independent of both. Three times now the error has been in what
was measured rather than in the measurement.

## 49. How the field measures recovery, and why our measurement was the weak link (2026-09-27)

Asked what other papers do to measure recovery. Gathered from today's four literature
sweeps (subagent reports with citations; the pattern is consistent across independent
sweeps, individual numbers would want verification).

    paper              primary recovery metric                        generative reasoning
    TransMLA           7-task MC avg (MMLU, ARC-e/c, PIQA,            none -- gsm8k is
                       HellaSwag, OBQA, WinoGrande)                   DEFINED in its
                                                                      lighteval/tasks.py
                                                                      and never launched
    MHA2MLA            same MC suite + LongBench                      none
    CARE               6-task MC via OpenCompass ppl scoring          none
    X-EcoMLA           9-task LM-Harness MC + LongBench               none
    Palu               WikiText-2/C4 ppl + MC + LongBench             none
    MOHAWK             WinoGrande, ARC-E/C, PIQA, HellaSwag, LAMBADA  none
    LoLCATs            MC + 5-shot MMLU                               none
    Llamba             ARC/PIQA/HellaSwag/MMLU/Lambada/WinoGrande     none
    SUPRA              MC + long-context QA                           none
    DroPE              ppl trajectories + 4 MC + NIAH + LongBench     none
    MatryoshkaKV       6-task MC (CPT stage)                          GSM8K, but only
                                                                      AFTER task LoRA SFT
    Mamba-in-Llama     AlpacaEval2 LC, MT-Bench, GSM8K               yes (hybrids only)

THE STANDARD IS: a 6-9 task LIKELIHOOD-RANKED MULTIPLE-CHOICE average, plus held-out
perplexity, plus LongBench whenever the claim involves KV compression, plus NIAH/RULER
for positional or context claims. Compression papers add throughput and memory.

### 49.1 This explains #46.3's resolution problem

MC scoring is DETERMINISTIC likelihood ranking -- no sampling -- averaged over 7 tasks
and tens of thousands of items. That is why those papers can report "-0.27%" or "93% of
the gap recovered" with confidence. We chose SAMPLED GENERATIVE EXACT-MATCH on ONE task
at n=1319, giving sigma ~1.4 points on a difference. We picked the highest-variance
measurement available and then could not resolve 2-point effects.

### 49.2 #26 was right for the wrong reason

#26 rejected HumanEval (n=164) for low power and concluded "rank arms on GSM8K". The
correct conclusion was "use a deterministic aggregate", not "use a bigger generative
benchmark". The whole family was wrong, not the sample size.

### 49.3 Two avoidable sources of variance in our own setup

  * TEMPERATURE 0.7 on GSM8K. Every paper above scores deterministically. Greedy
    decoding removes sampling noise at zero cost.
  * ONE task instead of an average over several.

### 49.4 What we lack

NO likelihood-ranked MC evaluation exists in this repo (grepped: no hellaswag, arc,
winogrande, piqa, obqa or mmlu anywhere), and lm_eval is not installed. A minimal MC
scorer is ~50 lines against the existing model-building machinery, cheaper than pulling
in lm_eval's dependency tree, and it is the ONLY way to compare against TransMLA's or
MHA2MLA's published numbers -- which we currently cannot do at all.

CONCLUSION: today's weak link was the MEASUREMENT, not the architecture. The arms were
fine; the instrument could not read them.

## 50. Greedy decoding is correct and our code was wrong to refuse it (2026-09-27)

bench_full.py refuses --temperature 0 with: "the Qwen3/3.5 cards explicitly forbid it
(repetition and degradation)". VERIFIED LOCALLY: models/qwen3.5-4b/README.md is 1,174
lines and contains ZERO occurrences of "greedy", case-insensitive. There is no
prohibition on the card we actually shipped against. Our own
ckpt/qwen3.5-4b-stageAB/generation_config.json has no do_sample and no temperature, so
HF's default applies -- do_sample=False -- meaning GREEDY IS THE MODEL'S OWN DEFAULT
and we have been overriding it to sample.

### 50.1 Where the warning actually lives

On the Qwen3 HYBRID cards and QwQ-32B, and there it sits on the THINKING-MODE bullet
(traces of 32k+ tokens). The non-thinking bullet omits it. It is on no base model, no
Qwen2.5 model, and nowhere in the Qwen3.5 family.

### 50.2 The convention, from primary sources

  * lm-eval-harness sets do_sample:false on ALL TEN gsm8k task variants; only the
    *-self-consistency ones sample, at T=0.2 with maj@k.
  * Cobbe et al. 2021: "a single low temperature (T=0) sample", and both T values
    "were chosen empirically to produce the best results" -- T=0 was OPTIMAL, not
    merely conventional.
  * Qwen's OWN math harness (QwenLM/Qwen2.5-Math/evaluation/sh/eval.sh) runs
    gsm8k/math at --temperature 0 --n_sampling 1, and Qwen ships do_sample:false in
    generation_config.json for every base model AND for Qwen2.5-Math-7B-Instruct.
  * lm-eval-harness PR #3620 settles explicitly that the TASK's greedy setting
    outranks a model's generation_config.json.
  * RedHat evaluates 4-bit W4A16 Qwen3 greedy on GSM8K (87.64 -> 85.97, 98.1%
    recovery), reserving sampling for AIME-length traces only. Closest analogue to us.

### 50.3 Two of my own arguments were wrong

  * "Short generations are safe": Holtzman measured 73.66% greedy repetition WITHIN
    200 tokens, and states the feedback loop holds "regardless of phrase length". The
    protective factor is PROMPT CONSTRAINT, not length. Wiher et al. 2022 measured the
    task-conditional version: <1% repetition on all directed tasks, and 8-shot GSM8K
    with fixed exemplars is directed.
  * "T=0.2 as a safe middle ground": Holtzman found "sampling with temperatures lower
    than 0.9 severely increase repetition" -- low-T over-truncates and MIMICS greedy.
    It is a comparability compromise, not a remedy.

### 50.4 The gap is small either way

min-p paper Table 14, 8-shot GSM8K, single sample: the BEST sampling config beats
greedy by +0.98 (Mistral 7B) and +0.15 (Llama 3.1 8B). A controlled Qwen3-32B pair
(lm-eval issue #3129) moved 0.001 between greedy and the card's recommended sampling --
the 50-point swings in that thread came from chat templates and thinking traces.

### 50.5 Also: our GSM8K was never comparable to the literature

We run ZERO-SHOT with a custom "#### <number>" instruction; the field runs 4-shot
(Qwen) or 8-shot CoT with its own exemplars and extraction regexes. Biderman et al.
2024 document prompt-format changes alone shifting scores >20 points. Our GSM8K is its
own benchmark. What we got RIGHT: capping generation and scoring no-match as wrong
matches the harness's "[invalid]" fallback, and our 768 cap is 3x its 256 default.

ACTION: switch to greedy, and report truncation + repetition rates alongside. NOT yet
applied, because switching mid-sweep would break comparability with tonight's numbers.

## 51. c0 mechanism arms: taps unresolved, conv best on perplexity (2026-09-27)

    arm                ppl@8192   GSM8K                 vs control
    c0-ungrouped150      9.093    1150/1319 = 87.2%     CONTROL
    c0-taps150           9.233    1160/1319 = 87.9%     +0.76p z=+0.82 p=0.46
    c0-conv150           9.082    (running)             best ppl of any arm

taps150: UNRESOLVED, exactly as #46.3 predicted. Note ppl and GSM8K DISAGREE again --
taps is 0.14 nats WORSE on ppl and 0.76 points BETTER on GSM8K, neither significantly.
The taps did train properly this time (14.6% of on-path RMS at the corrected 6e-4,
linear in LR), so this is a fair test of the mechanism, unlike the first attempt.

conv150's 9.082 is the best perplexity of any arm, marginally ahead of the control --
the user's depthwise-conv-on-down-and-up idea. Its GSM8K is pending.

### 51.1 MoL moved to the front, on request

Justified: #45.3 caps capacity mechanisms at <=2.1 points against ~1.4 paired sigma, so
the remaining arms chase noise, and taps confirms it. MoL's claim is different in KIND
-- parity at HALF the cache -- which this eval CAN establish, because the null
hypothesis is the one we want.
queue_v9 stopped (conv150's bench survived the parent kill); queue_v10 runs
mol4-latent (E=4, ungrouped 4096, same cache) then mol8-r256 (E=8, ungrouped 2046,
87.5% compression, the arm with the testable claim), then the deprioritised gate150 and
f2a2-150. Built cache/mla_groups_ungrouped_2048.json by halving each layer's rank,
preserving the allocation shape.
EXPECTATION SET IN ADVANCE: with copy init, #41.1 measured expert gradients at 0.889
correlation, so these may be exact nulls because the experts never differentiate. If so
the fix is the per-cluster init, not a different mechanism.

## 52. MoL never engaged; routing/differentiation literature, verified (2026-09-28)

### 52.1 What the trained MoL checkpoints show

    arm              pairwise cos(experts)  divergence  router rms (init 0.020)
    c0-mol4-latent        0.999596            0.48%          0.02104
    c0-mol8-r256          0.999865            0.40%          0.02111
EXPERTS ARE STILL IDENTICAL COPIES; ROUTER BARELY MOVED. Causes, all implementation:
  (a) LatentRouter computed NO load-balance loss, and mol_aux_loss() only collected
      from the x-routed MoLRouter -- zero balancing in both arms.
  (b) train.py guarded the mol_aux_loss call on a.mla_mol alone, so the latent variant
      could not have reached it even with (a) fixed. Both fixed and verified.
  (c) COPY INIT. Divergence 0.00477 ~= lr*steps = 0.0045: the experts moved exactly
      their displacement ceiling. #41.1 had already measured 0.889 gradient correlation.
CONSEQUENCE: with identical experts MoL's output equals plain MLA for ANY routing (the
exactness property verified at install), so mol8-r256's 75.2% IS ~plain MLA at
r=2046. That supplies the missing control: 75% compression 87.2% -> 87.5% compression
75.2%. HALVING THE CACHE COSTS ~12 POINTS WITH NO MECHANISM. MoL is UNTESTED, not
refuted.

### 52.2 Soft-to-hard annealing does NOT break symmetry -- verified from the PDF

A WebFetch summary of arXiv 2605.02124 (Rastegar, Meta, "Soft-to-Hard Routing in Sparse
MoE") claimed annealing "achieves faster convergence and better expert utilization than
hard routing from the start". THAT WAS FABRICATED BY THE SUMMARISER. The paper is a
THEORY paper (boundary-layer calculus, population squared loss, "synthetic diagnostics
included only as controlled checks"). Its section 6.1 says the opposite of the summary:
    "At exact symmetry, f1 = f2 and h(.; u, theta) is independent of u ... small random
    perturbations ... break this perfect symmetry and induce a small but non-zero expert
    contrast. The question is then whether the router amplifies this contrast"
With identical experts the router gets NO gradient -- exactly what 52.1 measured.
Annealing can only AMPLIFY an existing contrast. It is not a symmetry breaker.
LESSON: WebFetch summaries of PDFs can invent results; verify load-bearing claims
against the saved text.

### 52.3 Options, mapped to OUR constraints (top-1, causal decode, recomputable cache)

  CLUSTER-AWARE UPCYCLING, arXiv 2604.13508 (CVPR 2026). Essentially our per-cluster
    plan, independently published: spherical k-means on PCA-reduced (8x) activations,
    per-cluster WHITENED truncated SVD (Cholesky S S^T = X X^T, keep rank >= 95% energy
    and > half full rank), router = L2-NORMALISED centroids with NO temperature.
    Measured: lower pairwise expert similarity, lower routing entropy, no collapse.
    CAVEATS: gains modest (+0.2 on their retrieval/classification), grew over 1.3B
    samples on 64 H200s -- we train 150 steps. And they needed no temperature because
    they route on COSINE (normalised centroids and inputs), which bounds the logits;
    our raw-dot-product centroid router saturated (router gradient exactly 0, #43-era).
    ADOPT: cosine routing instead of the temperature hack.
  VARIANCE LOSS, arXiv 2505.22323: maximise per-expert routing-score variance across
    tokens. Works with top-1 and directly attacks the near-uniform router of 52.1.
    Coefficient not recovered from the fetch -- needs checking before use.
  ORTHOGONALITY LOSS, same paper: orthogonalises outputs of the MULTIPLE experts
    SELECTED for a token. With TOP-1 only one expert is selected, so it is DEGENERATE
    for us as written. Adaptation would compute all E decoder outputs densely in
    training (E x the up-projection) and orthogonalise those. Second priority.
  DEEPSEEK-V3 AUX-LOSS-FREE BALANCING, arXiv 2412.19437: a per-expert bias added to
    the SELECTION score only, nudged by observed load, excluded from the gating weight
    -- no interference gradient. Clean alternative/complement to the Switch loss.
  NOISE INJECTION: the cheapest symmetry breaker, and the mechanism 2605.02124 itself
    names. A baseline, not a strategy.
  EXPERT CHOICE, arXiv 2202.09368: balanced by construction, no aux loss -- but each
    expert picks its top-k tokens OVER THE SEQUENCE, i.e. it needs future tokens. NOT
    USABLE at causal decode, and our latent routing must decide per token at write
    time. Training with it and inferring with token choice would add a train/inference
    mismatch. SKIP.
  ReMoE / LapSum-SoftMoE / DSelect-k: variable experts per token. Compatible with the
    cache (still recomputable from the latent) but makes decode cost variable. Later.

### 52.4 Proposed next MoL arm

Latent-space cluster-aware init (per-region least-squares decoders, #44 measured -13%
at E=4 and -30 to -35% at E=8 at init) + COSINE router from normalised centroids +
variance loss + Switch or DeepSeek-bias balancing, optionally a short anneal ON TOP.
Test at 87.5% compression against the 75.2% no-mechanism baseline and the 87.2% target.

## 53. MoL rebuilt: cluster-aware init, cosine router, DeepSeek bias (2026-09-28)

User's spec: the Cluster-Aware Upcycling init (or adapted), cosine routing, NO aux loss
but DeepSeek's bias instead (no balancing coefficient), variance loss ONLY if it
measurably improves on top of the bias.

### 53.1 Implementation (src/mercurius/surgery/mol.py)

  ROUTER, mode="cosine": scores = exp(log_scale) * cos(c, w_e), log_scale learnable
    (init 10). Cosine bounds the logits -- the reason 2604.13508 needs no temperature
    and our raw-dot centroid router saturated to an exactly zero gradient.
  SELECTION = argmax(scores + bias); GATING uses the UNBIASED score (DeepSeek-V3).
  BIAS IS A Parameter, NOT A BUFFER: save_trainable() persists requires_grad params
    only, so a buffer would be dropped at save and eval would route WITHOUT the
    trained bias, silently. It enters only an argmax, so .grad stays None and AdamW
    skips it; it moves only through mol_update_bias(), once per optimizer step.
  LOAD is OVERWRITTEN per forward, not accumulated: grad checkpointing re-runs the
    forward in backward and an accumulator would double-count.
  CLUSTER INIT (mol_cluster_init): c = down(X); spherical k-means++ (best of 4) on c;
    router w <- normalised centroids; each expert's up_k/up_v <- ridged least squares
    from c to the TRUE K = X W_K^T, V = X W_V^T. A cluster below 1.5*r rows keeps the
    global fit. W_K/W_V are now stashed on LatentKV at construction (CPU, ~10 MB/layer,
    not in the state_dict) -- the latent cannot supply them (#35).
  RUNS ON THE FULLY BUILT STUDENT, just before evaluate(0), from data/calib_mix.txt.
    A separate calibration script through care.build() would have been WRONG: it
    hardcodes install_rope_dial(m, 0, "global"), i.e. NoPE hidden states, while line 1
    is c0. In-training calibration sees exactly what the arm trains on.
  LEGACY mode="dot" kept; retrieval_ab picks the mode from the keys (lin.* -> dot).

### 53.2 Verified on a toy LatentKV with clustered inputs

    rel error  plain 0.7332 -> global LS refit 0.2842 -> per-cluster 0.2413
    expert pairwise cos 0.752 (copy init 1.000); router/true-cluster agreement 100%
    bias: requires_grad True, grad None, present in the save set; aux loss None
    round trip both modes: 0 missing / 0 unexpected, max|dK| 0, bias restored exactly

### 53.3 DeepSeek's gamma is WRONG FOR OUR BUDGET

Rebalancing a 3.5x-skewed router, max/mean load:
    gamma   @50    @150
    1e-3    3.23   3.17     DeepSeek's value -- bias moves only 0.15 in 150 steps
    3e-3    3.17   2.89
    1e-2    2.85   1.74
    3e-2    1.76   1.01     <- default
    1e-1    1.02   1.02     fast but overshoots
The bias must travel a few LOGIT units, so gamma scales with the router scale:
~0.003 x scale. "No balancing coefficient in the objective" holds; the update RATE is
still a real hyperparameter at 150 steps. With cluster init the router STARTS balanced,
so the bias mostly guards drift.

### 53.4 A confound the design must control

The cluster init does TWO things: REFITS the decoders on data, and SPLITS them. In the
toy most of the gain was the refit (0.733 -> 0.284) not the split (0.284 -> 0.241). An
E=8 win could be a refit win any single decoder would get. CONTROL: the same init with
E=1, i.e. refit and no routing.

### 53.5 Arms, all at HALF cache (ungrouped 2046 = 87.5% compression)

    mol8-ca        E=8, cluster init, cosine, DeepSeek bias        the main test
    mol1-ca        E=1, same init                                  refit-only control
    mol8-ca-var    as mol8-ca + variance loss                      A/B for the var loss
References: no-mechanism 87.5% = 75.2% (#52, the inert mol8-r256); full-cache 75%
control = 87.2%. The user's goal is FIXED QUALITY AT LOWER CACHE: success is closing
the 12-point gap, not beating 87.2%.

## 54. A shared encoder keeps MLA's bottleneck; untied routers cannot find the pairing (2026-09-28)

USER CORRECTION: a shared encoder is "MLA with extra steps" -- every token is projected
onto the SAME r-dim subspace of x, and decoders only reshape those r numbers. Right. The
shared encoder came from #44 dropping the index, which traded away the mechanism's core
(different regions keep different subspaces of x) to save a few bits.

TOY, held-out [K;V] error, plain MLA 0.2548:
    tied pairs (decoder knows the encoder)          0.2225  -12.7%
    untied: independent decoder router on c         0.7528 +195%   purity 0.51
    untied + gauge rotation to distinct directions  0.7801 +206%   purity 0.35
    z-slice: shared routing slice k=2 of r=16       0.2257 -11.4%  no index
    shared encoder, 4 latent-routed decoders        0.2376  -6.8%
WHY UNTIED FAILS: each cluster's whitened SVD writes its own COORDINATE FRAME
(down_e = Vh_e[:r] L_e^-1), so a decoder fed several encoders' latents reads
incompatible coordinates, and cosine routing cannot tell the frames apart.
WHY PAIRING IS THE WHOLE GAIN: with a FIXED decoder U the optimal encoder is U^+ W for
EVERY input distribution -- the covariance drops out of the LS condition -- so routed
encoders buy nothing unless the decoder varies WITH them.

## 55. Tied pairs with a cached index (option 1) (2026-09-28)

Index cost: ceil(log2 E) bits per token per MLA layer against r bf16 values:
    87.5% compression (2046 values = 32,736 bits):  E=4 16 bits 0.05%, E=8 24 bits 0.07%
    worst layer r=94: 0.2%; stored lazily as a uint8 per layer, ~0.2% overall
User chose option 1 (tied); option 2 (decoder router reading (e, c), restoring
E_enc x E_dec mixing now that the decoder knows the frame) remains available.

IMPLEMENTATION: install_mol_routed(tied=True) creates E stacked (down, up_k, up_v)
copies AT INSTALL (so the optimizer sees them) plus a cosine router on x with the
DeepSeek bias; mol_tied_cluster_init fills them at step 0: spherical k-means on x, pair
e <- LatentKV.mol_factor(cov_e) -- the MLA constructor's own whitened-SVD math.
TOY: exact at install; E=1 reproduces plain EXACTLY (-0.0%); E=4/8 -12.7% held-out;
router/k-means agreement 100%; replay round trip clean.
Plain down/up are frozen and excluded from the "MLA latents trainable" re-enable loop
(they are unused by the routed forward; re-enabling only bloated checkpoints).

CALIBRATION FIX: random 1024-token windows across the WHOLE calib file (as CARE does).
The #53 version read calib_mix.txt from the start, and that file is concatenated BY
SOURCE (first 21% SWE chat episodes) -- the likely cause of #53's worse step-0 ppl.

GATE BEFORE ANY TRAINING: step-0 ppl@8192 vs plain MLA at the same cache, 12.664.
E=1 smoke separates "calibration sound" from "pairs help".

## 56. MoL measured properly: step-0 ppl, held-out offline fits, output-Fisher metric (2026-09-28)

### 56.1 Step 0, 87.5% compression (ungrouped water-filled 2046), c0
    plain MLA (stored CARE cov, NoPE-era sample)   ppl@2048 10.072  @8192 12.664  CE 2.539
    tied E=1 (same method, RECALIBRATED on the c0 student, random windows)
                                                    9.560           12.140        2.497
    tied E=8 (+3-bit index)                         9.091           11.506        2.443
Half of E=8's apparent gain over plain is RECALIBRATION (E=1: -0.042 nats), which every
MLA arm should get for free. MoL proper = E=8 vs E=1: -0.054 nats @8192 (-5.2% ppl).
MEASURED uncompressed (no MLA, same c0 student, step 0): CE 2.3237 @8192 (ppl 10.213),
2.0746 @2048 (7.961). (An earlier INFERRED 2.251 was wrong and understated MoL.)
Degradation @8192: plain-stored 0.215, E=1 recalibrated 0.173, E=8 0.119 (x0.69 vs
E=1; @2048 x0.73); plain at 75% (4094, stored cov) 0.071 -- rank doubling cuts
degradation to ~1/3, and E=8 closes 53% of the 87.5%->75% gap at equal cache. The
offline Fisher proxy predicted x0.82 for this ratio: it UNDERSTATES the real loss
effect, so its cache-at-equal-quality numbers (56.3) are conservative.

### 56.2 Held-out offline fit (experiments/mol_offline_fit.py; balanced [K;V] metric)
Fit on 131k vectors from fineweb_edu_long's first half; test on its second half and
WikiText. In-sample numbers (#43: ~50% cache) were badly optimistic. Held out, cache
at equal error vs plain@2046: E=8 cosine 18.4% / 14.2% (fw / wiki), best-of-E 21.8% /
18.0%; E=16 22.2% / 17.3%. Each doubling of E cuts error only 2-3%, i.e. worth
0.06-0.32 rank doublings -- a slow power law, the signature of a continuous
high-dimensional input distribution, not a union of few subspaces. Best-of-E routing
(argmin_e ||W x - U_e D_e x||, deployable: needs x and W only) adds 1-8 pts over
cosine, more on late layers. fp64 SVD was 8x slower than fp32 on this GPU with 1e-4
agreement; fits run fp32 on fp64-accumulated covariances.

### 56.3 OUTPUT FISHER (experiments/mol_fisher_fit.py). USER CORRECTION: the balanced
[K;V] error "doesn't really mean anything" -- it ignores k_norm, RoPE, q.k, o_proj and
treats the K/V balance heuristic as importance. G = E[g g^T], g = dCE/d[k;v] from
backward passes on 256 fit windows; dCE ~= 1/2 E[d^T G d]. The Kronecker weight
(input cov x output Fisher) is solved exactly by SVD(G^1/2 W L) (Manton et al. 2003).
Held out, 8 layers:
  * FISHER-FIT PLAIN vs CARE-FIT PLAIN: 32% / 35% cache at equal Fisher error; summed
    dCE proxy 0.0377 -> 0.0243 (x0.64); per layer x0.36 (L31) .. x0.85 (L7). An INIT
    effect: training optimises the same function class, so expect it to wash out
    (#25) -- but it is the correct baseline for any trained MoL comparison.
  * DURABLE MoL CAPACITY (Fisher-fit MoL vs Fisher-fit plain): E=8 x0.82 / x0.85 on
    the summed proxy, 14.5% / 11.3% cache at equal quality (cosine), 17.5% / 14.7%
    best-of-E; E=16 17.4% / 13.7% (cosine). Per layer E=8 x0.72-0.87.
  * Proxy calibration: RATIOS match step-0 ppl (E=8/E=1 predicted x0.82, measured
    x0.78); ABSOLUTE values are ~6x too small (layer-local, no compounding). Use it
    for comparisons only.
  * Pipeline: --mol-fisher-windows N (capture_kv_fisher + mol_factor(G=)); smokes
    queued. Captured on the model as built (compressed), unlike the offline fit.

## 57. Which latent structure? Top-1 wins; combinatorial structures buy nothing (2026-09-28)

experiments/mol_structures_fit.py: all variants closed-form (reduced-rank regression
in the Fisher metric), best-of-E routing, scored as decode(encode(x)) through explicit
cache codecs (decode sees only latent + indices + up-projections; verified exact).
Mean x dCE vs plain over layers 3/7/19/31, held-out fineweb:
    top1-8      0.776   8x params        resid-8x8    0.781   8x  (64 subspaces)
    top1-16     0.731   16x              resid-16x16  0.738   16x (256 subspaces)
    shared+8    0.844   ~4.5x            shared+16    0.802   ~8.5x
* RESIDUAL 2-STAGE TIES TOP-1 AT EQUAL PARAMS ON EVERY LAYER: stage 2 refines inside
  the region stage 1 chose; the useful variation is one "which region" decision, not
  independent factors. Combinatorial count is not capacity here.
* SHARED EXPERT: at fixed cache it is a constrained top-1 (no capacity, by the math);
  measured slightly WORSE per parameter too (shared+16 0.802 vs top1-8 0.776, similar
  params). DeepSeek's shared expert helps training/compute efficiency, not this.
* TOP-2 SUM: first run INVALID (worse than plain -- impossible when fitted: the top and
  next h directions of plain SVD reproduce plain). Cause: all dictionary pieces
  initialised from one top-1 fit, so every pair was redundant. Fixed: dictionary =
  resid stage-1 U stage-2 pieces, so top2sum-E <= resid-(E/2)^2 at equal params by
  construction. RE-FIT (logs/mol_structures_top2.json), mean x dCE, layers 3/7/19/31:
      top2sum-8  0.840 (4x params)   top2sum-16 0.774 (8x)   top2sum-32 0.730 (16x)
  vs top1-8 0.776 / top1-16 0.731 at the SAME params: a tie (0.001-0.002).
* CONCLUSION: every structure lies on ONE quality-per-parameter curve (spread
  0.002-0.007). 120-496 pairs, 64-256 residual combos and 8-16 whole pieces buy the
  same thing for the same weights: capacity here is the NUMBER OF PIECE PARAMETERS,
  not how they are combined. Choose on cost and training behaviour. Arm C for the
  trained comparison = top2sum-16 (won by 0.002, i.e. a coin flip; chosen as the
  structure most unlike top-1, the most informative test of whether TRAINING can use
  combinatorial structure that closed-form fitting cannot).
* "equiv rank" pricing (plain rank that matches MoL@r0) flatters MoL vs "rank MoL can
  drop to while matching plain@r0" (56.3's 15-18% for E=8). Use the latter for claims.

### 56.4 The Fisher-aware init FAILS in real perplexity (step 0, E=1, 87.5%)
    CARE fit, recalibrated    ppl@2048  9.560  @8192 12.140  top1 54.9%
    Fisher fit (pipeline)             10.481        12.583        52.4%   (+0.092 / +0.036 nats)
The offline proxy predicted x0.64 on summed dCE; the real loss got WORSE. So the
empirical-Fisher proxy cannot pick between fits, and the Fisher-fit rows of 56.3 are
UNCONFIRMED. CARE-fit MoL comparisons stand (real ppl: E=8/E=1 x0.69 of degradation).
Also: the pipeline G's absolute scale is ~50x the offline G's (per-layer plain dCE sum
1.86 vs 0.038 nats; real step-0 degradation 0.215). Different data (calib_mix vs
fineweb) and model state (compressed student vs uncompressed base).
Suspects, untested: (1) empirical Fisher != Hessian (Kunstner et al. 2019) -- low
gradient-variance directions still carry curvature; (2) G dominated by high-loss
outlier tokens (summed CE; calib_mix has code/chat); (3) damping 1e-3 too weak.
Trained arms A/B use CARE (queue rule: Fisher only if it beats CARE at step 0).

## 58. TRAINED MoL arms, judged on RETRIEVAL (2026-09-29)

USER DIRECTION: judge a KV-cache mechanism on retrieval / long-context tests, not plain
ppl. All arms 87.5% (ungrouped water-filled 2046), CARE init (Fisher lost, #56.4), 150
steps, c0, same data order; all routers LEARNED (cosine + DeepSeek bias, no aux loss).
  A  E=1 (plain MLA through the same path)          1x latent params
  B  top-1 E=8                                      8x
  C  grouped top-2 = resid 8x8, one learned router per group   8x
Every arm is REPLAY-CHECKED before benchmarking (experiments/replay_check.py: rebuild on
the trainer's NF4 base must reproduce the training eval CE; all three: diff 0.0000).

### 58.1 ppl (final eval)   CE @2048 / @8192
    A 2.0443 / 2.2608    B 2.0271 / 2.2537    C 2.0361 / 2.2571
    plain 75% (2x cache, c0-ungrouped150) 1.9710 / 2.2075
Step-0 MoL gains mostly WASH OUT in ppl (E=8: -0.054 nats at step 0 -> -0.007 trained).
C's step-0 lead over B (0.020) reverses after training.

### 58.2 RULER multi-needle (5 tasks x 4k/8k/16k x 50 samples, EM on 20), same samples
    value-token NLL (the retrieved content, formatting excluded), mean of 5 tasks:
                 4k      8k      16k       EM@16k
      A        0.0685  0.0748  0.1126     86.5%
      B        0.0678  0.0686  0.0912     92.5%    (-19% at 16k)
      C        0.0678  0.0687  0.1059     87.5%    (-6%)
THE MoL GAIN IS A LONG-RANGE RETRIEVAL GAIN, largest beyond the 8k training length, and
~20x larger in relative terms than the ppl difference suggested. Long ppl (4 books to
32k) agrees: B = A in the first 2k tokens (+0.2%), ~1% better from 4k to 32k.
C (combinatorial) keeps only a third of B's 16k gain: structure buys nothing offline
(#57) and generalises worse past the training length after training. TOP-1 E=8 IS THE
DESIGN TO CARRY FORWARD.
CAVEAT: these runs saved per-cell means only (no paired per-sample test). ruler.py now
saves per-sample values (idx, nll_s, nll_v_s, em_s); NLL-only reruns of A and B are
queued for a paired test. Seed replicates (--run-seed 1) of A and B are queued too.
Next: B vs the 75% arm (2x cache) on the same RULER -- the "same quality at lower
cache" question directly.

### 58.3 B vs PLAIN AT 2x CACHE (75%, c0-ungrouped150; replay diff 0.0000)
    value NLL mean of 5 tasks:   4k      8k      16k     EM@16k
      A  (87.5%)               0.0685  0.0748  0.1126   86.5%
      B  (87.5%, E=8)          0.0678  0.0686  0.0912   92.5%
      R75 (75%, plain)         0.0593  0.0548  0.0637   95.75%
B closes 8% / 31% / 44% of the A->R75 value-NLL gap (4k/8k/16k); 39% of total NLL at
16k. By task at 16k (total NLL): multikey_2 89%, multikey_3 85%, multikey_1 34%,
multivalue 5%, multiquery 2%.
ANSWER TO "same quality at lower cache": E=8 MoL is worth ~1.3x cache (closing ~40% of
a doubling = 2^0.4), i.e. ~20-25% cache saving at equal retrieval quality, for 8x
latent params (+66M, ~1.6% of the model). THREE INDEPENDENT ESTIMATES AGREE: offline
fit (E=8 ~ plain at 1.3x rank), step-0 ppl (~0.4 rank doublings), trained RULER.
It is NOT a doubling. The gain is concentrated in multi-key retrieval; 2x cache also
improves multivalue/multiquery, which MoL does not touch.
Control caveat: R75 trained a day earlier from the STORED-cov init (0.042 nats worse
at step 0 than A's recalibrated init) -- if anything R75's lead is understated.

### 58.4 FRAMING CORRECTION (user): the fair test is EQUAL compression
Whether MoL WORKS is B vs A at the same 87.5% cache, same init / path / data -- and it
does: RULER value NLL -19% at 16k, -8% at 8k, EM@16k 86.5 -> 92.5, long ppl ~1% better
from 4k to 32k; A's seed-1 replicate moves NLL by only 0.001-0.02 per cell, so B's
multi-key gains (0.04-0.08) are 2.5-80x the seed spread. 58.3 compared MoL against plain
at 2x cache, which answers a different question (how much cache MoL is worth) against a
single, unmatched yardstick (R75: older run, stored-cov init) -- a modest mechanism will
always lose to a doubling, and that is not evidence it failed. The right instrument for
"same quality at lower cache" is a MATCHED LADDER of plain arms (E=1, same init and
path) at ~85/83/81%, reading off the compression at which plain equals B.

### 57.1 CORRECTION (user, 2026-09-29): the "top-2 sum" tested in #57 was the WRONG design
#57's top2sum gave each chosen piece its OWN rank-r/2 latent and added the two
reconstructions -- i.e. top-1 at half rank, twice (split rank, like resid). The
intended top-2: ONE latent at the FULL budget r written by two encoders and read by two
decoders:  c = (D_a + D_b) x,  K = (U_a + U_b) c  -- every pair gets its own rank-r map
(cross terms U_a D_b), so E experts give E + C(E,2) full-rank maps for top-1's params
and cache (+1 index). #57's conclusion ("structures all tie top-1") does NOT cover it.
Also confirmed: in MoL every expert has the FULL rank r (arm B: down_w (E, r, d)); only
resid (by design) and the mistaken top2sum split the rank.
Caveat for the correct design: per-cluster fits live in DIFFERENT latent frames, so
D_a + D_b / U_a + U_b are meaningless at init unless the experts are gauge-aligned
(orthogonal Procrustes onto a common frame; average rather than sum to keep scale).

## 59. PROPOSAL (not implemented): online in-context decoder -- RLS / delta-rule MLA (2026-09-29)

STATUS: design only. Implement + test after the operating-point arms (#58). Origin: user,
asking whether an online Oja-style compressor could replace MoL's fixed expert set.

WHY. #56-#57 put MoL on a slow power law (x0.94 per doubling of E; every structure --
top-1, residual, split top-2, merged full-rank top-2 -- on ONE quality/param curve): a
FIXED set of pieces covering a continuum. A decoder fitted to THE CURRENT CONTEXT is not
a fixed set. Causal LLM => no "sequences", only a growing prefix: the design must be
online, incremental, with a bounded state and consistent decoding of every past token.

### 59.1 From Oja to what we need
Oja: online PCA, min E||x - W W^T x||^2, W <- W + eta (x - W y) y^T, y = W^T x. Assumes
stationarity, TIED encoder/decoder, optimises FUTURE samples, and DISCARDS the past.
We need, at every step T, every past token decodable from its code written ONCE at time j
(k_j is gone after writing):   min_{M_T} sum_{j<=T} ||k_j - M_T c_j||^2,  bounded state.
Minimal modifications:
  1. UNTIE encoder and decoder.
  2. Least squares over the WHOLE PREFIX, not a stochastic step on the current sample:
     M_T = S_T P_T^-1,  S_T = sum k_j c_j^T (out x r),  P_T = sum c_j c_j^T (r x r).
     Per token = recursive least squares = the DELTA RULE WITH GAIN P^-1:
     M <- M + (k_t - M c_t) c_t^T P_t^-1.
  3. STATE = sufficient statistics (S, P), not a basis. Decoding every past code with the
     CURRENT M_T is LS-optimal for the whole prefix: no history, no replay, no per-chunk
     bases; old tokens' reconstructions only IMPROVE as statistics accumulate. The residual
     each code discards is not lost -- it enters S via k_t c_t^T.
  4. COLD START = prior from the trained static decoder U: S_0 = U P_0, P_0 = lambda I ->
     M_T = (U P_0 + S_T)(P_0 + P_T)^-1 (Bayesian blend; lambda = trust in the prior).
  5. Forgetting (gated S, P as in Gated DeltaNet) optional: tracks drift but degrades old
     tokens; default none.
  6. The LS decoder is metric-INVARIANT (same M under any quadratic metric), so a
     query-aware metric (whitening / Fisher) only matters for the ENCODER (what is kept).
Rejected alternatives considered on the way: per-chunk stored bases (basis costs r x out
per chunk; only amortises at chunks >> out), re-expressing all cached codes per update
(O(N r^2) per step), append-only orthogonal directions (works, but strictly less general
than keeping S, P), plain Oja/delta with D_0 + D_t only (the update history is the
discarded residual: not reconstructible from the endpoints).

### 59.2 Architecture: MLA whose up-projections are learned IN CONTEXT
  * cache per token: only c_j (r values). Encoder: our trained static down projection
    (variant A), or c_t = M_{t-1}^+ k_t, projection onto the current decoder, so the
    SUBSPACE itself adapts (variant B).
  * state per layer: S (out x r) + P (r x r) for keys, a second memory for values
    (~0.6 M values/layer at r=256: fixed, ~7% of the cache at 32k tokens; same kind of
    state as our 24 GDN layers).
  * attention stays latent: u_T = M_T^T q_T (a DeltaNet-style scan), then ordinary
    absorbed latent attention of u_T against cached codes; values: attend over codes,
    then decode once with the value memory.
  * training: parallel, causal, with the chunked delta-rule (WY) machinery the GDN
    layers already use.
  * open: partial RoPE is applied after decoding -> decoupled RoPE key for the rotated
    64 dims (DeepSeek-style) or keep them on the static path. P^-1 needs a ridge.

### 59.3 Test plan (offline first, ~30 min to write)
What decides it: how much of the discarded residual is LINEARLY PREDICTABLE FROM THE KEPT
CODE WITHIN ONE CONTEXT, beyond the global decoder. On real 8-16k documents, K and V
separately, at our per-layer ranks: for each prefix length T fit M_T (RLS with the U
prior), decode all j <= T, compare to static U. Encoder variants A and B; sweep lambda;
also report error on the NEWEST vs OLDEST tokens (consistency over time). Break-even:
beat MoL E=8 (~x0.8 error / ~1.3x rank) at equal cache, counting the state as fixed
overhead. If it does, build it in the pipeline (arm vs A and B at 87.5%, RULER-judged).

### 57.2 The CORRECT merged top-2 measured (experiments/mol_top2merged_fit.py)
c = 1/2 (D_a + D_b) x, K = 1/2 (U_a + U_b) c, full rank r, E=8 (28 pair maps), same params
/ cache (+1 index) as top-1 E=8; both fitted by the same GD procedure from the same
Procrustes-aligned top-1 start; exact best-pair routing. Held-out fineweb, vs plain:
    layer        3       7       19      31
    top-1      -13.3%  -9.7%  -13.2%  -11.9%
    top-2m     -12.8%  -9.4%  -12.4%  -10.7%   (pairs used 22/28/26/14 of 28)
Merged top-2 trails top-1 slightly on EVERY layer (WikiText agrees). Five structures now
sit on one quality/param curve. The ceiling is the data (a continuum covered by a fixed
set of pieces), not the combination rule -> #59 (context-adaptive decoder) is the lead.

### 58.5 Seed noise on RULER (A replicate, --run-seed 1; replay diff 0.0000)
    mean value NLL     A s0    A s1    spread   B       B - mean(A)
      4k             0.0685  0.0690  0.0005   0.0678  -0.0010  (2.0x spread)
      8k             0.0748  0.0735  0.0013   0.0686  -0.0056  (4.3x)
      16k            0.1126  0.1110  0.0016   0.0912  -0.0206  (12.7x)
Per task @16k (value NLL, A s0 / A s1 / B): multikey_1 .108/.105/.062, multikey_3
.090/.090/.041, multiquery .094/.098/.074 (B better); multivalue .256/.242/.235 (even);
multikey_2 .015/.020/.043 (B WORSE, and A's seeds agree -- a real trade-off). The
long-context retrieval gain of MoL at equal cache is far outside training noise. B's own
replicate (queued) completes the check. ppl seed spread: 0.001-0.002 nats.

### 58.6 EQUAL-CACHE test at 80.8% (3151 values/token): MoL's retrieval gain VANISHES
Arms: E=8 (B81) vs E=1 (A81), same recalibrated init / path / recipe; replay 0.0000.
ppl trained: E=1 1.9879 / 2.2170, E=8 1.9867 / 2.2174 -- identical (step 0: E=8 -0.062
nats @8192, fully washed out; at 87.5% a -0.007 residue survived).
RULER, PAIRED per-sample over identical samples (first per-sample runs), E=8 - E=1:
  answer NLL, multi-key: -0.015 (z -4.5) / -0.008 (z -2.0) / -0.014 (z -3.5) at 4/8/16k
  answer NLL, all 5 tasks, all cells: -0.004 (z -2.5); multivalue +0.010 at 16k (z +2.4)
  VALUE NLL (retrieved content), all cells: +0.002 (z +1.1); @16k -0.002 (z -0.8)
=> at 81% E=8 adds only non-value answer-token confidence; the retrieved content is the
same as plain MLA. Contrast 87.5%: value NLL @16k -19% (-0.021, ~13x A's seed spread;
per-cell means -- paired reruns queued).
CONCLUSION: MoL's retrieval benefit is a LOW-RANK phenomenon. With 54% more rank, plain
MLA trains its way to the same retrieval. The #58.3 "E=8@81% ~ plain@75%" was mostly the
rank, not MoL (the user's equal-cache correction, 58.4, was exactly the right control).
REVISED RECOMMENDATION: ~81% (5.2x) -> plain MLA (MoL not worth 8x latent params);
87.5% (8x) -> top-1 E=8 (real long-range retrieval recovery, +66M params). MoL is a
tool for AGGRESSIVE compression, not a free lunch at moderate compression.

### 58.7 Next: generation tests, then settle the recipe (user, 2026-09-29)
HYPOTHESIS (user, to test when NoPE returns): MoL may matter more for NoPE. With RoPE,
retrieval leans on position; without it, retrieval is pure content matching in K, which
loads the latent far more -- exactly the regime (small latent) where MoL helped. Plain
MLA at 75% and 81% already look decent under c0.
GENERATION TESTS (queue15), greedy (#50; bench_full.py's greedy refusal removed -- it
cited a Qwen card claim that does not exist), --no-think, max-new 768, truncation
reported: GSM8K then HumanEval + MBPP on plain 75 / 81 / 87.5%, MoL E=8 81 / 87.5%, and
the original. If plain and MoL are indistinguishable on generation: RECIPE = plain MLA,
CARE whitening recalibrated on the student, ungrouped (one latent per layer), water-filled
across layers, c0; compression chosen from these results. USER CONFIRMS BEFORE THE RUN.

## 60. Long-run preparation: objective, calibration, decontamination (2026-09-29)

OBJECTIVE (kept): reverse KL with TAID (prob space) + ce_beta 1.0 x EXCESS CE on the
true token, CE(y,s) - CE(y,t) -- its gradient is ordinary CE; the teacher offset only
makes the log read as excess. (--ce-beta help text corrected: it still described the
pre-fix divergence-dependent data term.) --lm-weight is a redundant extra CE, off.
MATH (policy 12.2, already settled): OpenMathInstruct-1 GSM8K-derived rows with
is_correct (MIT problems, Apache-2.0 Mixtral solutions) + procedural problems
(mathematics_dataset) with teacher-written solutions only, `generator` field on every
row. Still to BUILD, and its share of the mix to decide.
CALIBRATION ON THE TRAINING MIX, SEQUENTIAL (--mla-calib N --mla-budget B,
src/mercurius/calibration/mla_seq.py): windows from the actual training sources in
their proportions; covariances on the fully built c0 student; CARE water-filling; layer-
by-layer conversion with inputs re-collected from the partly compressed model; saves a
covs file + groups JSON for exact replay. 4B test (24 windows x 512): one-shot CE 2.2142
-> sequential 2.2025 (-0.012) at the same allocation; input-cov drift 5% (L7) -> 11-13%
(L19+). Plain water-filling on this data gives 171-343 per layer vs the old
retrieval-weighted 94-394 -- adopt it (principled; allocation washes out, #25).
data/calib_mix.txt, used by every previous calibration, is NOT the training mix and has
no builder script.
DECONTAMINATION (experiments/decontam.py): word 13-grams vs GSM8K test, HumanEval,
MBPP, WikiText eval, MC suite (951k eval grams, same loaders as the benchmarks).
fineweb_edu_long 3/3522 flagged (wikitext 2, mc 1); pilot_mix 1/1973 (humaneval 1);
synth_recall 0/880. Clean copies: data/fineweb_edu_long.decon.txt,
data/episodes/pilot_mix.decon.jsonl -- the long run trains AND calibrates on these.
Re-run on the math data once built.
LOOPED REASONING added to roadmap 4.5 as research-first (user); not scheduled.
OPEN FOR THE USER: math share + token budget; conv MLA (test ~4.5 h, or drop -- evidence
so far: ppl tie, GSM8K +1.44 p=0.12).

## 61. Long-run DATA and BUDGET decided (user delegated, 2026-09-29); conv MLA dropped

CONV MLA: DROPPED for now (user). Evidence never exceeded a ppl tie + GSM8K +1.44 p=0.12.

MATH -- text chain-of-thought only. VERIFIED on 400 streamed rows: 393/400 (98%) of
OpenMathInstruct-1's GSM8K-derived correct rows are CODE-INTERPRETER style (<llm-code>
block, fake tool output, \boxed{}); training on them teaches "running code" with
invented outputs, since there is no interpreter at inference. Policy 12.2 accepted the
LICENCE; the FORMAT is disqualifying. Sources, all policy-clean:
  A. GSM8K TRAIN, the HUMAN-WRITTEN solutions (MIT; calculator annotations <<...>>
     stripped) -- no model anywhere. 7,473 problems.
  B. teacher-written text CoT for the same GSM8K train problems (Apache-2.0 35B-A3B,
     conditioned on the known answer, kept only if the final answer matches) -- style
     diversity.
  C. OpenMathInstruct-1 GSM8K rows, is_correct AND text-only (~2%), <=2 per question.
  D. procedural problems (google-deepmind/mathematics_dataset, Apache-2.0) with teacher
     solutions conditioned on the computed answer -- volume and variety beyond GSM8K.
Every row: `generator` + `source` + `license` fields; packed into multi-problem chat
sequences (~4k tokens) so the episode sampler's min length holds; decontaminated
against GSM8K test and every eval set (experiments/decontam.py) before use.

BUDGET: 100M tokens (~3.6 days at ~325 tok/s, teacher-bound), eval checkpoints at
10M / 25M / 50M / 100M to read the recovery curve (stop early if flat; keep the LAST).
MIX by TOKEN share: long-document text 70% (FineWeb-Edu docs >= 8k tokens,
decontaminated: 30M now + ~40M to fetch, ~1 epoch), agent episodes 20% (pilot_mix.decon,
9.6M -> ~2 epochs), math 10% (~10M from A-D, ~1 epoch). Synthetic recall corpus OUT
(never tested under c0).
TO BUILD: math data (A-D); ~40M more long FineWeb-Edu tokens; a weighted multi-source
sampler (token-share targets -> per-step probabilities from each source's mean length),
shared with the calibration sampler; the final recipe script -- presented to the user
for confirmation before starting.

## 62. Compression choice, retrieval evidence: plain 75% vs 81% (2026-09-29)

Matched plain arms (E=1, same recalibrated init / path / recipe), replay 0.0000.
ppl trained: 75% 1.9746 / 2.2046, 81% 1.9879 / 2.2170 (0.012-0.013 nats, ~10x seed noise).
RULER, PAIRED per-sample (81% minus 75%; + = 75% better):
  answer NLL  +0.010 (z 4.0) / +0.012 (z 7.6) / +0.020 (z 10.1) at 4/8/16k; all +0.014 (z 11.8)
  value NLL   +0.001 (z 0.3) / -0.001 (z -0.5) / +0.008 (z 4.7);      all +0.003 (z 2.4)
  MoL81 minus plain 75%: answer +0.011 (z 6.3), value +0.004 (z 2.7)
Means @16k (value NLL | EM): plain75 0.0625 | 97.75%  plain81 0.0703 | 95.75%
  MoL81 0.0681 | 95.25%  ORIGINAL 0.0652 | 97.0%.
PLAIN 75% IS AT THE ORIGINAL'S RETRIEVAL LEVEL; 81% loses ~8% value NLL at 16k only (and
answer NLL everywhere), and MoL at 81% does not recover it. Leaning 75% for the "fixed
quality" goal; the greedy generation tests (queue16: P81, P75 first) decide whether the
81% retrieval cost shows up in generation.

### 62.1 Greedy GSM8K: 75% and 81% are INDISTINGUISHABLE
Greedy, no-think, max-new 768, full 1,319 (rescored from the per-item generations; matches
the harness): plain 75% 1154 (87.5%, 36 truncated), plain 81% 1152 (87.3%, 35).
McNemar: 62 vs 60 discordant, z +0.18. RECOMMENDATION (pending HumanEval/MBPP): 81%
(3151 values/token, 23% below 75%) -- reasoning unchanged, ppl +0.012 nats, the only
measurable cost is long-range retrieval beyond the 8k training length (16k value NLL
+0.008, EM -2 pts), which a 100M-token run should partly recover. 75% if retrieval at
the original's level is a hard requirement. Revise if the code tasks separate them.

### 62.2 DECIDED (user): 75% compression, exactly 4x
Chosen for the clean 4x factor (communication) with a small quality difference either way
(#62.1: GSM8K indistinguishable; 75% slightly better on ppl and long-range retrieval, at
the original's retrieval level). Recipe budget: --mla-budget 4096 (= 16384 / 4, exact),
water-filled by the sequential calibration on the training mix. Code tests (HumanEval +
MBPP, greedy) run P75 then P81 as a confirmation; the remaining arms (MoL 81/87.5, plain
87.5, original) follow and are cancelled if the long run starts first.

## 63. Resume made complete and VERIFIED; long run started, leg 1 (2026-09-30)

GAPS FOUND in the resume path (all fixed in train.py):
  1. the episode/math SOURCE stream (_eg) was re-seeded on resume -> a resumed run
     replayed the same episode/math draws. Now saved/restored with the other streams.
  2. global torch / CUDA / python / numpy RNG and the on-policy coin: not saved. Now saved.
  3. loss history and realised source counts reset on resume. Now saved.
  4. calibration recomputed on resume -> GPU float noise could flip a water-filling tie,
     change a rank, and break the load. Now REUSED from the saved files (same settings
     and data -> same init).
  5. no planned split: --stop-at-step N keeps the full --steps horizon (LR, TAID,
     sampler), force-writes the resume file at N plus a labelled snapshot, exits without
     the final eval / final adapters.
  6. data could change between legs silently: the resume file carries a (size, mtime)
     fingerprint of every data file and refuses on mismatch.
  7. resume file written in place: now write-then-rename (an interruption during the
     save cannot destroy the only resume file).
  8. evaluate(0) re-ran on resume and logged a trained model as "step 0": skipped.
  9. torch>=2.6 weights_only load refused the (own) file with RNG states: loads with
     weights_only=False (caught BY the test).
Already correct: optimizer moments, OneCycleLR position (+ guard on --steps mismatch),
TAID state, trainable weights, window-offset and length-mix streams.
VERIFICATION (12-step runs, same settings): R uninterrupted; S1 stop at 6; S2 resume
from S1 to 12; R2 a second uninterrupted run. S1 = R to float noise on steps 1-6 (step 6
loss 0.5339 vs 0.5346, same sources, same TAID t); S2's step 7 is the SAME math draw as
R's (source stream continues). Final eval CE @2048/@8192: R 2.0664/2.3379, R2
2.0646/2.3315, S2 2.0673/2.3324. Median relative weight diff: R vs R2 8.4%, R vs S2
9.1% -- a resumed run differs from an uninterrupted one only as much as two
uninterrupted runs differ from each other (GPU nondeterminism, amplified on zero-init
params). RESUME IS EQUIVALENT.
ALSO SETTLED: MBPP greedy 75% 169/257 vs 81% 165/257, McNemar z +0.63 -- all three
generation benchmarks indistinguishable (GSM8K +0.18, HumanEval -0.38, MBPP +0.63).
LONG RUN, LEG 1 STARTED (experiments/run_long_75.sh --go): 15,750-step schedule
(100M tokens), stops at 7,875 (~1.8 days, an eval step); leg 2: --go --resume; an
unplanned interruption in leg 1: --go --resume-leg1. Math final: 1,678 packs (A-C + 1,849
procedural rows). ONE teacher (:8077, 8 slots) serves training ONLY for the duration.

## 64. Leg 2 = QAT for the deployed format: 4-bit weights + 4-bit KV (2026-10-01)

User: QAT was missing from the recipe; deploy 4-bit weights and a 4-bit KV cache; leg 2
starts with it. Leg 1 (bf16 adapters on an NF4 frozen base) is unchanged.
  * Already QAT-like before this: every frozen base Linear trains as NF4 (QLoRA), so the
    base quantization error is trained around. NOT seen until now: the rounding of the
    MERGED weights (VeRA delta, per-head q maps, MLA latents, the 755 M GDN-2 gate
    weights), of the tied embedding, and of the KV latent.
  * Format: NF4 b64 (double-quant stats) for weights -- bitsandbytes' own kernels, so
    fake-quant == deployed weight bit for bit; and NF4(W_nf4) == W_nf4 exactly (tested),
    so only what the adapters moved gets re-rounded (int4 would re-grid everything).
    KV: the MLA latent, int4 symmetric g32 + fp16 scale (4.5 bits/value), optional
    fixed orthogonal rotation folded into down/up. Gates: NF4 (or int8), measured.
  * Gradients: value from the quantized merged weight (detached) + the trainable path
    as a zero-valued residual on x.detach() -> exact input gradient, STE parameter
    gradient in rank space, no dense weight gradient. Unit-tested vs a dense reference
    (bf16-level agreement); folds exact to 1e-7. No parameter added/renamed: leg-1
    resume files load unchanged. models/qat.py; --qat* flags in train / ruler /
    bench_full / replay_check (build(qat=...)).
  * Boundary (experiments/leg_boundary_qat.sh): PTQ sweep (qat_ptq_sweep.py: body W4,
    gates bf16/int8/NF4, KV g16/32/64 x rot none/orth, embedding) chooses KVROT and
    GATEBITS; 25-step QAT smoke on a copy; then leg 2 with --qat. Leg 1's endpoint is
    kept as ckpt/resume-c0-long75-leg1end.pt. A resumed --qat run evaluates once at its
    start step = the PTQ cost, which QAT then has the decay half of the schedule to
    recover.
  * Deployment memory (single sequence, weights + KV + 52 MB recurrent state):
        ctx                         8k     32k    128k    256k  (GB)
        original bf16 / bf16 KV    8.73    9.54   12.76   17.05
        original 4-bit / f16 KV    2.49    3.30    6.52   10.81
        original 4-bit / 4-bit KV  2.30    2.52    3.43    4.64
        student 4-bit (NF4 gates)  2.62    2.68    2.91    3.21
    Student weights are LARGER than the original's at equal bits (2.55 vs 2.17 GB):
    the KDA/GDN-2 lifts added 755 M channel-wise gate parameters. KV is 2.25 KiB/token
    vs 32 KiB (14.2x; 4x of it from MLA, the rest from 4-bit). Crossover vs the
    original at 4-bit weights + f16 KV: ~12.5k tokens; vs 4-bit weights + 4-bit KV: ~55k.

### 64.1 GDN-2 gates deploy in their EXACT factored form, so QAT leaves them bf16
Reference layers (fla): KDA and GDN-2 make the DECAY gate low-rank (hidden->128->4096)
and the output gate low-rank; GDN-2's erase/write gates are dense. Ours are all dense
(tiled lifts, 755 M params). Spectrum of the trained gates (step 3150, plain Frobenius,
72 matrices): the trained delta is 7-9% of the base norm and VeRA spreads it over the
whole 1024-rank basis -- relative error median 0.05-0.06 at rank 128 (half the learned
delta's energy lost), 0.03-0.04 at 256, 0.014-0.019 at 512. So reference-style rank 128
would discard much of what training did, and dense NF4 adds ~9% unstructured rounding
error per gate, more than the whole delta. But the EXACT form is tiny: each gate is a
tiled base (32 unique rows) + diag(b) B diag(d) A with the SHARED bf16 A (1024x9216) and
B (9216x1024) (+ the decay's rank-32 LoRA). Deployed that way: ~0.06 GB instead of 0.39
(NF4 dense), zero error, and fewer bytes read per token (u = A x is shared by the three
gates of a layer -- same input, same A slice -- and B[:4096] is read once for three
vectors: ~0.33 GB/token over 24 layers vs 0.39 dense NF4), and half the FLOPs.
=> leg 2 runs --qat-gate-bits 16. Weights 2.22 GB (vs original 2.17 at NF4); at 1M
tokens 4.69 GB total.

## 65. Leg 2 adds 32k windows at constant tokens; teacher patched to serve them (2026-10-02)

Mid-run benchmarks (step 7875, before leg 2; user: review before continuing):
  RULER 4/8/16k paired vs R75-plain (150 steps): answer NLL +0.010 (z 4.2) better;
  VALUE NLL -0.007 / -0.009 / -0.024 at 4/8/16k (z -2.1 / -2.8 / -3.9) worse; vs the
  original, value NLL level to 8k, -0.023 at 16k (z -3.3); EM no detectable difference
  (8 vs 10 discordant). Only the 16k effect survives correction for ~8 tests.
  GSM8K greedy 1171/1319 (88.8%) vs P75 1154 (87.5%): +1.3 +/- 0.9 pts, exact McNemar
  p 0.18 (79 vs 62 discordant); truncations 51 vs 36.
Hypothesis (user): 8k-only training under RoPE degrades beyond 8k. Leg 2 therefore mixes
LONG CORPUS WINDOWS at the SAME mean length, so tokens/step, total tokens, the step
schedule and every source's token share are unchanged: corpus windows
4096:0.318 / 8192:0.629 / 32768:0.053 (mean 8192.0); 32k windows = 15.0% of all tokens.
--length-mix-spec (train.py); the old 8k cap does not apply to --teacher-server (its KL
is chunked from hidden states).
Teacher: stock llama.cpp cannot return pooling-none hidden states for a prompt beyond
one micro-batch (embeddings force n_batch = n_ubatch), and the 35B-A3B MoE path fails
above ~12k tokens per micro-batch (12288 ok; 14336 illegal memory access; 24576 /
33792 assert in mmid.cu). Patched llama-server (~/llama.cpp-split, worktree of the same
commit 4ceb171; opt-in LLAMA_SPLIT_EMBD_NONE=1): a causal pooling-none prompt is split
into 8k micro-batches through the KV cache, per-token outputs concatenated in order,
no prefix reuse. experiments/check_teacher_split.py, teacher-head KL on 512 positions:
noise floor of the kernels themselves (8k vs 12k batch shape, unpatched) 0.00140;
patched split 12k vs unpatched 12k 0.00173 (top-1 98.1%); patched 8k == unpatched 8k
exactly; 32k request returns all rows. Served as the ONE teacher (8077, 2 x 33792).
Eval speed-ups the same day (models/fast_infer.py, --merge-eval, RULER early EM stop):
checked equivalent (experiments/check_fast_infer.py); fla's Triton conv UPDATE broke
decode on the real cache and is NOT used.

## 66. KV latent QAT = TurboQuant-MSE (no QJL), not a group-absmax int grid (2026-10-02)

User: TurboQuant without QJL is very good -> rotation-based QAT in its style, not an
arbitrary rotation in front of an int grid. PTQ sweep (#64, step 7875, CE @8192 over W4):
int4 g32 no rotation +0.0039, any rotation in front of the absmax int grid WORSE (+0.0097
at g32) -- expected: a uniform absmax grid wastes levels on the near-Gaussian rotated
coordinates. TurboQuant-MSE: fixed random orthogonal rotation per layer (folded into
down / up_k / up_v, free at inference), fp16 norm per token per layer, unit vector
quantized coordinate-wise with the Lloyd-Max codebook of the rotated coordinate's exact
density f(x) ~ (1-x^2)^((r-3)/2) (data-oblivious, one codebook per (r, bits)). 4.03
bits/value at 4 bits (vs 4.5 for int4 g32); 3-bit = 3.03. Codebook checked against the
Gaussian Lloyd-Max levels (0.128 ... 2.733 at 4 bits, x sqrt(r)). models/qat.py
--qat-kv-quant tq (STE); leg-2 default (KVQ=tq). PTQ cost on the leg-1 model measured
before leg 2 (logs/qat_ptq_tq.json).

### 66.1 Before leg 2: long-context evals, TQ PTQ, QAT probes (2026-10-03)
RULER 32k/64k (25 samples/task, EM 10), mean EM: original 100 / 100, R75 91.5 / 74.0,
L1 (step 7875) 73.5 / 64.5. Value NLL paired L1 vs R75: -0.051 (z -3.5) @32k, -0.053
(z -1.9) @64k; L1 vs original -0.064 (z -3.6), -0.162 (z -7.1). The long run LOST
retrieval beyond its 8k training length relative to the 150-step arm -- the case for the
32k windows (#65).
PG-19 NLL by position (12 books, paired per book), delta vs original: L1 +0.003 / +0.013
/ +0.011 / +0.015 / +0.019 / +0.034 for 0-2k / 2-4k / 4-8k / 8-16k / 16-32k / 32-64k; R75
+0.010 ... +0.031. L1 beats R75 up to 32k (-0.004..-0.008, t -2..-3), loses beyond
(+0.003, t +2.5). Both students' deficit GROWS with position.
TQ PTQ (over W4, gates bf16, CE @2048 / @8192): int4 g32 +0.0054 / +0.0045; TurboQuant
4-bit +0.0065 / +0.0122; 3-bit +0.0218 / +0.0233. So before QAT, TQ-4 costs +0.008 nats
more than int4 g32 @8k at 0.47 bits/value less.
QAT probes (3 steps each, TQ on): 8k-only peak 13.8 GiB; 32k-only peak 26.5 GiB, a 32k
step ~140 s (~235 tok/s). The boundary smoke's OOM was a 32k draw under the 24 GiB cap:
leg 2 runs with --mem-cap-gb 40. Each probe spent ~80 min in its start-of-leg eval: QAT
decode re-quantized every weight per token -> no-grad forwards now cache the deployed
weight per parameter version (models/qat.py _cached; equal to the grad path bit for bit).

### 66.2 Diagnosis of the long-range retrieval loss (2026-10-03)
RULER @32k value NLL (paired): step 0 (same init, untrained) 0.078 == R75 0.075 (z -0.4)
-> the CALIBRATION / INIT is not the cause. The loss arrives EARLY in training (step 1575
0.109, 4725 0.120, 7875 0.126) and is concentrated in the multi-KEY tasks (many similar
distractor keys; multikey_3 EM 30% vs R75 80%), not multi-value / multi-query: lost key
DISCRIMINATION among distractors, growing with length.
PG-19 next-token NLL meanwhile IMPROVES with training at every position, beyond 8k too
(step 0 +0.125 vs original at 32-64k -> L1 +0.034): "8k training breaks everything past 8k"
is false; precise long-range retrieval and general long-context LM move in opposite
directions.
Parameter drift: per-head query maps ||R-I||/||I|| 0.44 @1575 -> 0.99 @7875 (R75 0.11;
own LR 1e-3 = 33x the dense rate); MLA 1.7-3.5%; GDN static decay frozen (unchanged).
L1's R routes 37% of the mass reaching the SLOW RoPE pairs (17-31, < 90 deg over 8k) from
other dims (R75 1%).
Post-hoc ablations on the step-7875 model (RULER @32k vNLL / PG-19 delta 32-64k):
  R = I                  0.095 (+0.031 vs L1, z +1.9) / +0.55   -- co-adapted: LM breaks
  slow pairs protected   0.125 (z +0.1)               / +0.061  -- slow-pair mechanism FALSIFIED
  all rotary protected   0.325                        / +0.91   -- rotary mixing load-bearing
R is implicated (reset recovers ~60% of the gap) but not separably, and not through the
slow pairs. Decisive test (66.3): train WITHOUT per-head q from the same init, same data,
seed, sampler and schedule, to step 1575; compare RULER @32k / PG-19 with L1 @1575.

## 67. BUG: per-head query maps did not commute with RoPE -> fixed; long run restarted (2026-10-03)

PerHeadQ (surgery/perhead_q.py) maps q_h -> R_h^T q_h inside q_proj, i.e. BEFORE q_norm
and before RoPE rotates the first 64 of 256 head dims. Its justification ("a map on q is
a per-head key, foldable into the query = MLA absorption") assumes nothing
position-dependent between the two; under partial RoPE (dial c0) the score is
q^T R Rot(Delta) k, which equals a per-head key map q^T Rot(Delta) W k for all Delta only
if R commutes with the rotations -- the reason MLA keeps RoPE out of absorption
(DeepSeek, TransMLA). c0-long75 step 7875 measured: relative commutator of R's rotary
block 0.42 / 0.49 / 0.53 at Delta 100 / 8k / 64k; 12.8% of R's mass mixes rotary and NoPE
dims. So the maps re-routed query content across RoPE frequencies, fitted only on offsets
<= 8k -- consistent with #66.2 (strict recovery within 8k, multi-key retrieval loss beyond,
early onset when R moves fastest, partial recovery on reset).
Also pre-RoPE: trainable q_norm / k_norm per-dim gains; unequal gains within a rotary pair
do not commute with its rotation (moved ~0.3%; fixed the same way).
FIX: project after every optimizer step onto the RoPE commutant -- per rotary pair
(i, i+32) a 2x2 [[a,-b],[b,a]], no cross-pair and no rotary<->NoPE mixing, NoPE block
free (project_rope_commute_; commutator 1e-18, idempotent, I -> I); q/k norm gain CHANGES
tied per pair, original gains kept (tie_rope_norm_deltas_). The saved R always satisfies
it, so checkpoints replay without harness changes. --phq-rope-commute --rope-tie-norms.
Other pre-RoPE paths audited: VeRA on q_proj and the MLA up_k are maps from x / the latent
INTO q / k (ordinary fine-tuning of W_q / W_k, position-agnostic, what any recovery
does); RoPE is applied after the latent reconstruction (c0 keeps all 32 frequencies,
identical to the original); GDN layers have no RoPE.
RESTART c0-v2 (experiments/run_v2.sh), budget by Monday: 6300 steps / 40M tokens,
OneCycle over them; steps 1-1575 = leg-1 recipe + the fix (A/B vs c0-long75 @1575: RULER
32k value NLL 0.109, untrained 0.078); 32k windows from 1576 (#65); QAT from 3151 (#64,
#66: NF4 weights, TurboQuant-4 KV, bf16 gates, NF4 embedding), evaluated before/after
installing it at 3150. The no-R control (#66.2) was stopped at step 950 to free the GPU.

### 67.1 The map belongs AFTER q_norm; long-window mix from step 1 (user)
The #67 projection made R itself commute with RoPE, but R still sat in q_proj, BEFORE
q_norm, whose original per-dim gains (1 + w) differ strongly within rotary pairs (ratio
p10 0.73 / p90 1.31, worst 132x): the transform reaching RoPE is D R^T, which commutes only
for a D-conjugated R. Moved: install_per_head_q_post registers R on the q_norm module
(`q_norm.R`, no existing name changes) and applies it to the NORMALIZED query, right
before RoPE -- there "commutes with RoPE" is exactly "is a per-head key map". Checked:
identity no-op bit-exact, per-head application, commutator 6e-16 after projection.
Replay: build() installs the post map when a checkpoint has q_norm.R. QAT: the post map is
not folded into q_proj (it cannot be: q_norm is in between); it stays a bf16 per-head
256x256 map (8.4 M params), or is absorbed key-side, which commuting makes legal.
c0-v2 uses --per-head-q-post --phq-rope-commute --rope-tie-norms, and (user) the 32k
length mix FROM STEP 1 -- giving up the clean step-1575 A/B for long-range supervision
throughout.

## 68. Absorbable MLA (decoupled RoPE, TransMLA RoRoPE); THE run restarted as v3-absorb (2026-10-03)

Checked against the papers (user): DeepSeek-V2 decouples RoPE because W_UK cannot be
absorbed through it ("matrix multiplication does not obey a commutative law") -- a shared
RoPE key k^R (d_h/2) outside the latent; TransMLA's RoRoPE applies, per frequency, ONE
orthogonal U_l across heads, identical on re and im (Eq. 19: commutes with RoPE), keeps
RoPE only on the first resulting head and moves the rest into the NoPE latent; it reports
short-context benchmarks only. MHA2MLA keeps top-r RoPE subspaces per head and joint-SVDs
the NoPE K/V; LongBench evaluated. None places a frequency-mixing learned map between q
and RoPE (our #67 bug).
c0 (RoPE inside the latent) is correct but NOT absorbable: decode rebuilds every cached
token's rotary key, ~9x the attention FLOPs of absorbed MLA at long context. User:
3.6x cache with absorbed decode beats 4x without -> surgery/mla_rope.py, --mla-rope-decouple N:
  * k_norm folded exactly: per-dim gain (1+w) into every key row; per-head 1/rms from the
    frozen original W_k, cached (4 scalars/token/layer).
  * RoRoPE from the calibration covariance: U_l = principal axes of
    Wre Sigma Wre^T + Wim Sigma Wim^T over the 4 KV heads (deterministic signs).
  * top N=1 component per frequency: exact RoPE key (64/layer, trainable k_rope); the rest
    lose RoPE and join the gained NoPE rows + W_v in ONE CARE latent -- same whitening,
    water-filling (decoupled_spectra) and sequential calibration (mla_seq n_rope).
  * forward: one GQA attention on Q' = [rot(U q) | U q | q_nope], K' = [rot(k_rope) | k_res |
    k_nope_g] / rms_g, D^-1/2 scaling.
  Verified: N=4 at full rank reproduces the stock attention (3e-4, fp32); absorbed scores
  (query folded through W_UK against the cached latent + cached RoPE key + cached rms) ==
  expanded forward (2.8e-7). Real model, RoPE energy kept exactly by component 0: 68-79%
  of the rotary key energy (rotary = 17-44% of key energy); N=2 would keep 84-91% at 3.18x.
  Cache: 4096 latent + 512 RoPE + 32 rms = 4640/token = 3.53x.
QAT bug found on the way: TurboQuant's rotation was FOLDED into the latent's NF4 weights,
mixing the whitened-SVD factors' very unequal scales before blockwise NF4 (stress test:
latent error 2.2x vs 0.2). Now weights are NF4 as they are and the r x r rotation runs in
bf16 at write/read (absorbed: folds into the r-dim query). Likely part of TQ's worse PTQ
number in #66.1. The RoPE key is TurboQuant'd too (own fixed rotation, quantized pre-RoPE).
Run: experiments/run_v2.sh, tag v3-absorb -- leg-1 recipe (GDN-2 with factored gates kept
bf16 under QAT, ScaleNorm, train-norms, ALL-VeRA 1024, reverse KL + TAID + CE, dial c0
frequencies, sequential CARE calibration on the training mix) + #67 post-norm commuting
query maps + tied q/k norm changes + 32k length mix from step 1 + QAT from 3151 (NF4
weights, TurboQuant-4 latent and RoPE key, NF4 embedding). 6300 steps / 40M tokens.

### 68.1 v3-absorb crashed at step ~110: SDPA math fallback; fixed, resumed
OOM allocating 64 GiB = the full 32k x 32k x 16-head score matrix: transformers' SDPA
wrapper passes enable_gqa=True, which only the FLASH kernel accepts, and flash caps the
head dim at 256 (ours 448 q/k / 256 v) -> MATH kernel. The decoupled forward now calls
SDPA directly with the KV heads expanded (memory-efficient kernel: takes 448/256 with a
causal flag or a mask; checked fwd+bwd at 32k, 6.0 GiB for the layer), explicit
bottom-right mask for multi-token continuation on a cache. Exactness test unchanged
(2.9e-4). Resumed from the step-100 resume file (no loss beyond ~15 min).
Start-of-run: step-0 CE @2048 / @8192 2.0931 / 2.3556 vs c0 step 0 2.1014 / 2.3925 (not
equal cache: 4640 vs 4096 values/token).

### 68.2 TurboQuant rotations: fixed dense Haar-random, saved with the model (user prompt)
TurboQuant's guarantee needs a random orthogonal rotation (coordinates of the rotated unit
vector follow the Beta density the Lloyd-Max codebook is built for); randomized Hadamard
is the fast approximation used in practice, but needs power-of-2 dims -- our latents are
579/451/390/427/579/543/471/656, and padding would add CACHED coordinates. A dense rotation
costs r^2 ~ 0.2-0.4 M MACs/token/layer at write (absorbed decode: folded into the r-dim
query once per step) -- negligible vs ~4 B MACs/token. Kept: fixed, dense, Haar-random
(seeded QR, sign-fixed) per layer for the latent and the RoPE key. They are now SAVED: the
first QAT install writes ckpt/tq_rotations_<key>.pt (matrices, seeds, Lloyd-Max codebooks);
every later install regenerates, verifies bit-equality, and uses the saved ones. Tested
end to end through install_qat (save -> load -> identical outputs, fwd/bwd ok). Applies to
the running v3-absorb at its QAT install (step 3151): qat.py is first imported there.

### 68.3 Memory squeeze at step ~1075: teacher prompt cache; controlled restart
System MemAvailable fell to 3 GiB (unified memory: the trainer's GPU allocations count
against RAM). The patched llama-server had grown to 48 GiB RSS -- llama.cpp's default
host PROMPT CACHE (--cache-ram 8192 MiB) filling with 32k-window prompts that are never
reused (every window distinct; prefix reuse is off for split pooling-none tasks) -- plus
20 torch-inductor compile workers (~0.4 GiB RSS each) for one compiled norm. Fixed without
loss beyond ~1 min of steps: TeacherServer now retries for up to 10 min (was 3 attempts /
12 s, which made any teacher restart a training crash); stopped the trainer right after
the step-1100 resume save; teacher restarted with --cache-ram 0 (same patched binary and
slots); resumed with TORCHINDUCTOR_COMPILE_THREADS=1. After: teacher RSS 12.5 GiB, 62 GiB
available, no compile workers.

### 68.7 v3-absorb final (step 3900, annealed QAT): results and the 4-bit cost
In-training eval (deployed 4-bit form): CE 1.9743 / 2.2089 @2048 / @8192; replay exact.
Decomposition (replay variants of the final weights, CE @2048 / @8192):
  fully 4-bit (NF4 weights, TurboQuant-4 latent + RoPE key, NF4 embedding)  1.9743 / 2.2089
  NF4 weights only (bf16 KV, bf16 embedding)                                1.9770 / 2.1997
  no quantization at all (same weights)                                     1.9860 / 2.2149
  step 3150, bf16, before QAT                                               1.9543 / 2.1815
-> 4-bit KV + embedding cost ~+0.009 @8k; most of the gap is the weights (+0.018 vs 3150).
   QAT (575 steps, LR already annealing) recovered only ~30% of the PTQ cost (+0.038 ->
   +0.027 @8k). The weights co-adapted to NF4 (worse when unquantized). LESSON: start QAT
   much earlier (~40% of training, LR still high) or keep the most sensitive weights at
   8 bits; the 4-bit KV cache (TurboQuant) itself is cheap.
RULER 32k (deployed 4-bit): EM 79.5%, value NLL 0.098 -- vs v3@3150 bf16 0.081 (z -2.1),
L1 final bf16 0.126 (+0.028, z +2.3 in favour of the 4-bit v3), original 0.062.
PG-19 vs original: +0.026 / +0.034 / +0.040 / +0.039 / +0.048 (0-2k ... 32-64k) -- the bf16
profile shifted up ~+0.03: the 4-bit cost is uniform, not long-range.

### 68.8 Final benchmarks vs the ORIGINAL (greedy, same settings) (2026-10-05)
The 2026-09-24 "original" generation run was SAMPLED (T=0.6, before #50) -- not comparable.
Re-run greedy, no-think, max-new 768, NF4 base + bf16 KV (experiments/eval_orig_greedy.sh):
  original  GSM8K 1210/1319 (91.7%)  HumanEval 128/164 (78.0%)  MBPP 179/257 (69.6%)
  v3 final, fully 4-bit (eval_v3_bench.sh; GSM8K at batch 32, code at 16):
            GSM8K 1178/1319 (89.3%)  HumanEval 123/164 (75.0%)  MBPP 166/257 (64.6%)
  paired v3 vs original: GSM8K 44 vs 76 (p 0.004), HumanEval 16 vs 21 (p 0.51),
  MBPP 18 vs 31 (p 0.09).
  v3 vs L1 (bf16): GSM8K p 0.57, HumanEval p 1.0, MBPP p 0.87 -- level;
  v3 vs P75 (bf16): GSM8K 77 vs 53 (p 0.04, v3 better), code level.
=> the deployed 4-bit model keeps most of the original's reasoning/code ability, with a
small, consistent loss (significant on GSM8K) in line with the uniform +0.03-nat 4-bit
cost (#68.7). The lesson stands: start QAT earlier.
