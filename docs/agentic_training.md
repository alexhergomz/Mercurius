# Agentic training: what it needs that plain distillation does not

Our recovery objective (reverse KL + TAID + excess CE) was designed for text
distillation. Agent trajectories are not text: most of their tokens are
environment output that the model must READ and never WRITE. Training them like
text is a category error, and we were making it.

## 1. Mask tool output from BOTH loss terms -- NOT just CE

Measured on our own corpus:

    role          tokens    share
    tool         950,234    84.7%   file contents, grep results, search output
    assistant     93,723     8.4%   the behaviour we want
    system        43,740     3.9%
    user          34,573     3.1%

    per-episode supervised fraction after masking:
      verified_122b.jsonl   3,944 tokens ->  1,220 (30.9%)
      diffs.jsonl           7,622 tokens ->    696 ( 9.1%)

So an unmasked loss spent roughly twelve times more gradient teaching the model
to reproduce file contents than to decide what to do. On the diff tier it was
worse: 91% of the gradient taught the model to reproduce the BUGGY SOURCE FILE it
had just been shown, for a tier whose entire purpose is "produce this fix".

**I argued the divergence term was fine unmasked** -- reverse KL against the
teacher on tool-output text being "ordinary distillation on text the teacher also
saw". **That was wrong, and the field is unanimous against it.** Tool outputs are
PREFILL CONTEXT: the environment inserts them, the model never generates them, so
supervising their prediction supervises a behaviour that does not occur. All three
papers doing agentic distillation with a KL objective exclude observation tokens
from the divergence term by design:

  * Structured Agent Distillation (arXiv:2505.13820) -- separate KL terms for
    REASON and ACT spans; observations excluded from both, "as they encode
    deterministic feedback from the environment rather than agent behavior"
  * SOD (arXiv:2605.07725) -- tool-observation tokens excluded from the per-step
    divergence
  * Distilling LLM Agent into Small Models (arXiv:2505.17612, Qwen2.5-32B ->
    0.5B-7B, close to our scale) -- observations excluded from the loss

### The one real ablation (2505.13820, ALFWorld, 340M student, Table 5)

    segmented spans, observations excluded   56.3% success / 71.5% CoT match
    token-level baseline                     52.1%          / 68.1%
    flat, no span segmentation               48.2%          / 60.4%
    RANDOM span masking                      45.9%          / 57.7%

Random masking is WORSE than no masking. The lesson is "mask correctly", not
"mask". Getting the spans wrong is actively harmful.

### Codebase equivalents, if we want to cross-check our implementation

  * TRL: `SFTConfig(assistant_only_loss=True)`, via chat-template
    `{% generation %}` markers
  * LLaMA-Factory: `train_on_prompt=False` (the default), plus `mask_history`
    for last-turn-only
  * Axolotl: `roles_to_train: ["assistant"]`, `train_on_eos: turn|last|all`, and
    `message_field_training_detail` -- the only stock tool that can mask WITHIN a
    message, i.e. `<think>` separately from `<tool_call>`

None of them support a distillation KL term, so we extend rather than copy.

## 2. Off-policy costs more than expected -- the strongest argument for on-policy

Our trajectories come from the teacher, not the student. Measured penalties:

  * **SOD** (arXiv:2605.07725), agentic/tool-use, Qwen3-0.6B student:
    off-policy SFT **8.97%** -> on-policy distillation **20.04%** -> step-adaptive
    **24.22%** average. That is ~2.7x relative from going on-policy.
  * **GKD** (arXiv:2306.13649): on-policy > mixed > off-policy across
    summarisation, translation and maths; ~+17pp GSM8K (T5-Base).
  * SOD's Proposition 1: a single erroneous tool observation of length m makes the
    divergence grow SUPER-LINEARLY under off-policy mismatch. Long trajectories
    with one bad observation are where off-policy hurts most -- which is exactly
    our shape.
  * Naive on-policy destabilises at 1.7B scale without adaptive reweighting.

We already do the cheapest mitigation (rejection sampling, as SWE-Gym and
ToolLLM do). On-policy remains deferred, but this is a much bigger number than
the reason it was deferred for.

## 3. Train on recovery from bad observations, do not filter it out

**FireAct** (arXiv:2310.05915) measures robustness to corrupted tool output:

    tool output replaced with "None":  ReAct degrades 33.8%, FireAct-tuned 14.2%
    random observation corruption:     ReAct degrades 28%,   FireAct-tuned  5.1%

Training on recovery roughly halves-to-fifths the degradation. That supports the
correction/bridging machinery we built (mercurius/data/correction.py,
rationalize.py) and argues against discarding trajectories that hit tool errors.

Note SWE-Gym filters differently per component: success-only for POLICY data,
success AND failure mixed for VERIFIER training. Filtering is not universal --
it depends what the data is for, which is the rollout store's whole premise.

## 4. Other agentic-specific levers

  * **Compaction**: ToolLLM caps tool output at 1024 tokens -- compress
    unimportant JSON keys, then hard-truncate. Directly relevant at 84.7%.
  * **Turn weighting**: sparse and mostly binary. LLaMA-Factory's `mask_history`
    (last turn only) is the only stock option; SOD has the only principled
    per-step reweighting (weights decay under high student/teacher divergence and
    recover on realignment). Agent-FLAN weights capability TYPE, not turn
    position (reasoning : retrieval : understanding = 1 : 0.25 : 0.75). No
    standard continuous turn-position discount exists.
  * **Keep `<think>` in the loss.** No direct ablation exists, but Agent-FLAN
    shows halving reasoning-labelled data costs 1.1 T-Eval points, the largest
    drop of any capability they tested.
  * **REASON/ACT split**: 2505.13820 uses separate normalisation per span type,
    `L = lambda_r * L_CoT + lambda_a * L_Act` with both lambdas 1.0. Worth trying
    once basic masking lands.

## 5. Gaps in the literature, one of which we could fill

  * **No controlled 2x2 of masked-vs-unmasked CE against masked-vs-unmasked KL on
    tool tokens.** That is precisely our question, and our setup can answer it.
    Possibly publishable.
  * No ablation on masking `<think>` specifically.
  * Nothing found on training an agent to STOP rather than loop -- a targeted
    search returned zero hits. Our own measured failure mode was the model
    exploring and then not emitting an answer, so this gap is ours to fill too.
  * "ETO" (trajectory-level DPO for agents) could not be verified; do not cite.
