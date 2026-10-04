# Candidate improvements

Ordered by expected value against what has actually been measured, not by
novelty. Every entry must have a **function-preserving initialization**: this
model is carved out of a trained checkpoint and there is no budget to retrain
from scratch, so any change that cannot start as an exact identity is a change
that starts by destroying the thing being improved.

## What the measurements say

Four facts constrain the whole list.

1. **Single-item retrieval is intact.** RULER S-NIAH 1/2/3 at 4k, 8k and 16k:
   100% exact match and gold-span NLL 0.099 for the original, the 283.88M dense
   arm and the 4.60M VeRA arm alike. The surgery — 18 of 24 layers to linear
   attention, 4x KV compression, NoPE — costs nothing here.
2. **Multi-item retrieval is the defect.** Multi-needle EM falls from 98.8% to
   79-85%. Worst cases: multiquery 97.5% -> 42.5% at 16k, multivalue 100% ->
   55%. One fact is fine; several interfere.
3. **Adaptation budget is not the cause.** 283.88M dense scores 83.3% and 4.60M
   of VeRA scores 85.2%. A 62x difference in trainable parameters produces no
   advantage, so the constraint is architectural, not capacity.
4. **Teacher-forced likelihood and free generation disagree.** On multivalue the
   converted arms have BETTER gold-span NLL than the original (0.331 vs 0.348 at
   16k) and far worse EM. The model assigns high probability to each gold answer
   and then loses the thread emitting four of them in sequence. That is exposure
   bias, not a retrieval failure.

5. **Capacity is not the constraint, and three measurements say so.** 283.88M
   dense scored 83.3% multi-needle where 4.60M of VeRA scored 85.2%. LoRA rank 64
   was negative when tested. And GDN-2's 110K trainable gate parameters improved
   NLL on 24 of 24 retrieval cells. Any claim that an arm underperformed "because
   the adapter was too small" needs to answer these first.

Historically the largest single lever was data, not architecture: long documents
plus document-aware sampling was worth -2.22%, more than any architectural change
tried -- against LR at ~1%, rank accumulation at 0.4%, and rank-64 and
rsLoRA-at-mixed-ranks both negative. Training is 150 steps over 0.041 epochs of a
30M-token corpus. Nothing in the mix has ever required holding several facts at
once, which is exactly and only the capability that is broken.

---

## Tier 1 — targets a measured failure

### 1.0 Run the objective that was actually specified
**Confidence: high that it changes the result. Cost: one run. NEVER RUN.**

The intended objective is TAID + reverse KL + CE, combined in nats. Every arm to
date ran `divergence=forward, taid=False, ce_beta=1.0` -- forward KL plus CE.
The specified objective has not been run once, on any arm:

| run | divergence | taid | ce_beta |
|---|---|---|---|
| allvera | none | False | - |
| allvera-ce | forward | False | 1.0 |
| gdn2vera | forward | False | 1.0 |
| gdn2phq | forward | False | 1.0 |

All three components are implemented and verified, and none has executed:

- **reverse KL**, checked numerically: equals KL(student||teacher) exactly,
  non-negative, finite against the smoothed target, and exactly zero when the
  student equals the teacher in all three modes with the data term on.
- **TAID**, which walks the target from the student's OWN distribution toward the
  teacher as lam goes 0 -> 1, so the target is always reachable. lam = 1 is
  bit-identical to pure distillation. NOTE: lam here is a linear ramp; the
  paper's (arXiv:2501.16937) is adaptive. Still unreconciled.
- **CE mixed into the target** rather than added as a second loss:
  q = (1-w) p_teacher + w delta_y. CE is exactly forward KL against a one-hot,
  so a separately weighted CE would mean something different in each mode, and
  KL(p_s || one-hot) is infinite -- reverse cannot take a hard target at all.
  Mixing makes beta = 1 meaningful instead of a scale hunt.

Why this is first rather than an architectural change: the measured
"CE trades retrieval for perplexity" result (allvera+data: best perplexity
15.962, worst multi-needle EM 79.3%) was produced under FORWARD KL, which is
mode-covering -- it forces the student to spread mass wherever the teacher has
any. That is the wrong pressure for retrieval, whose correct output is peaked and
nearly deterministic. Reverse KL is mode-seeking and is the principled choice
when the student cannot represent the teacher, which is exactly our case: linear
attention, 4x compressed KV, NoPE. So the single most informative run available
is the objective that was asked for.

Screen as three arms on the GDN-2 base: reverse+CE, reverse+CE+TAID, js+CE.


### 1.1 Synthetic multi-item recall in the training mix  [MEASURED: findings 0.5 -- the only mechanism to improve BOTH perplexity and retrieval; weak beyond the distances the corpus teaches. Corpus now teaches 2k/4k/8k/16k/32k; running at a matched 20% share so distance is the only variable]
**Confidence: high. Cost: low.**

The model has never been trained on anything that requires retaining several
associations. Mix MQAR-style data (Arora et al., arXiv:2312.04927) and
multi-needle examples into the corpus: k key-value pairs planted in a long
document, several queried at the end. This is the standard remedy in the
linear-attention literature and it addresses fact 2 directly rather than
hoping an architectural change transfers.

Identity init: none needed, it is a data change. Risk: over-fitting to the
synthetic format, which is why it should be a small fraction of the mix and
NOLEX/RULER must stay held out in distribution.

### 1.2 On-policy distillation (GKD)  [findings 0.5b -- the first attempt failed because rollouts were generic prose. Now anchored at restatement headers and verified firing (findings 0.6); under test]
**Confidence: high. Cost: medium.**

Fact 4 is exposure bias, and the known fix is to train on sequences the student
generates rather than only on teacher-forced ground truth (Agarwal et al.,
generalized knowledge distillation). Current training is entirely teacher-forced,
so the model is never asked to recover from its own mistake — which is exactly
what emitting four magic numbers in a row requires.

Identity init: the objective changes, not the weights. Cost: generation inside
the training loop; a small on-policy fraction is enough to matter.

### 1.3 Linear-attention state as summary tokens for the MLA layers
**Confidence: medium-high. Cost: low-medium.**

Each linear layer carries a recurrent state S of shape (d_k, d_v) that is
already a compressive summary of everything seen. The softmax layers cannot see
it; they only see whatever the linear layers chose to write into the residual
stream. Read m summary vectors out of S with learned query directions and
prepend them to the MLA layers' keys and values.

Closest published: Infini-attention (Munkhdalai et al., arXiv:2404.07143) pairs a
compressive memory with local attention; recurrent memory transformers prepend
memory tokens. What is different here is that the memory already exists and is
free — no second mechanism to train.

Why it should help fact 2: interference is a property of reading a fixed-size
state through a lossy linear readout. Letting full attention address the state
directly gives the model a second, non-interfering path to the same content.

Identity init: gate the summary tokens' contribution to zero.

### 1.4 Widen up_k so every query head gets its own key
**Confidence: high. Cost: very low. SCHEDULED.**

The MLA conversion reads its output widths off the original k_proj/v_proj, so it
kept GQA's 2 kv heads and only compressed. Eight query heads therefore share two
key subspaces. TransMLA's actual claim is that GQA's replicated K/V is
rank-deficient and a latent lets each head have its own key at the same cache
cost. Four heads forced to share one key is interference by construction.

Cost 0.524M/layer x 6 = 3.1M, cache unchanged, FLOPs unchanged — attention
already materializes 8 replicated heads, this only stops forcing them equal.
Identity init: repeat each group's rows, which is what repeat_kv does downstream.

**Measured: the first attempt tested nothing.** R ended 0.42% off identity (max
off-diagonal 0.0017) after 150 steps, because it sat in the dense group at 3e-5
where AdamW's displacement ceiling is 0.0045 per element. Its +0.4 EM was a null
result from an unexercised mechanism, not evidence against the idea. R now has
its own group at --phq-lr 1e-3 (ceiling 0.15). The VeRA rate 1e-2 was rejected:
its ceiling of 1.5 exceeds the identity diagonal R starts from, so it can erase
the identity outright.

**How TransMLA does it, if 1e-3 also fails to move R** (arXiv:2502.07864).
They merge all GQA key heads into one and expand back, giving each head its own
up-projection W^UK_i initialised as a zero-padded identity SELECTOR -- identity
in that head's own slice, zeros elsewhere -- reproducing the GQA replication
exactly. The new freedom lives in the up-projection from the latent, not in a
transform on q. The two are algebraically equivalent
(q^T (W_h c) = ((W_h)^T q)^T c) but not equivalent to optimise.

Two caveats before treating that as a fix. TransMLA reports no special learning
rate, parameter group or warmup -- but it fine-tunes on 6B tokens, about 5,000x
our 1.2M, so "no LR tuning needed" may just mean "enough steps that it does not
matter". And Adam's step is scale-free per parameter, so a zero-initialised block
does not automatically travel further than a perturbed identity: the binding
constraint here is the step budget, not the parameterisation. If 1e-3 does not
move R, the honest next lever is more steps, with TransMLA's per-head
up-projection as the structural alternative to try alongside it.

Note on the factorization, stated carefully because I got this wrong once.
What justifies CARE here is a measurement on this model, not a citation:
the whitened factorization gives -5.02% summed activation error at an identical
4.00x KV ratio (findings, CARE rank allocation). Plain SVD minimises WEIGHT
error, ||W - W_hat||, when the quantity that reaches the residual stream is
activation error, ||X W^T - X W_hat^T|| -- a different objective (SVD-LLM / CARE).

On TransMLA specifically: arXiv:2502.07864v3 describes PCA on activations from a
calibration set, but v1 (11 Feb 2025) is the version this pipeline was built
against, and its abstract does not state the decomposition. So "TransMLA uses
plain SVD and CARE improved on it" may well be right for v1 while v3 has since
moved to an activation-aware method. Do not cite either as settled without
reading v1's method section. The whitening stays either way, on the measurement.

### 1.5 RETRACTED: "the GDN-2 gates need more capacity than VeRA"

Claimed on the basis of a flat KL trajectory and flat perplexity (15.956 vs
15.962). The RULER result refutes it: those same 110K trainable parameters
improved gold-span NLL on 24 of 24 cells and moved multivalue@16384 by +20
points of exact match. 110K parameters did that. Whatever limits the remaining
gap, it is not the size of the adapter on the gates.

This is the same error as the earlier "dense KDA is required" claim, made the
same way -- reading a capacity conclusion off a training-loss curve that cannot
see the capability in question.

### 1.5b Periodic merge-and-re-randomise for VeRA
**Confidence: medium. Cost: low.**

VeRA's real constraint is not parameter count, it is that A and B are a FIXED
random subspace: the update can only rescale directions it was handed. Merging
diag(b) B diag(d) A into the base and RE-DRAWING A and B each cycle accumulates
independent subspaces, giving an effective rank of cycles x rank at one cycle's
parameter cost. That attacks the binding constraint rather than the nominal one.

ReLoRA's scaffolding is already here (--relora-every, --relora-warmup,
merge_and_restart with jagged LR re-warmup) but merge_and_restart handles only
LoRALinear; the VeRA merge is not implemented. Measured value of rank
accumulation with LoRA on this model was +0.4%, so expectations should be modest.

### 1.6 Distil the HIDDEN STATE, not the distribution -- both terms, in nats
**Confidence: medium-high on cost, unproven on quality. Cost: low. PROPOSED.**

D13 cached the teacher's final hidden state and noted that only the lm_head
matmul survives (2.5 of 54 GFLOP/token). This removes that too, by expanding both
loss terms around the teacher's hidden state. With delta = h_s - h_t (or
P h_t for a narrower teacher), mu = W^T p_t and F = W^T(diag(p)-pp^T)W:

    KL(t||s)    ~=                      1/2 delta^T F delta
    excess CE   ~=  (mu - w_y)^T delta + 1/2 delta^T F delta

Verified numerically (V=4000, d=64) against exact terms: relative error 0.00% at
|delta|=0.001 through 0.16% at |delta|=1.0, for BOTH terms. So the objective
moves entirely into hidden space and STAYS IN NATS -- ce_beta keeps the meaning
it has today rather than becoming a scale hunt, which is the same reason D12's
excess-nats formulation was chosen.

What it buys: no softmax over 248,320, no logits tensor, no lm_head in the
training graph. The linear term (mu - w_y)^T delta is exactly the pressure to
BEAT the teacher on the true token, so the data term survives intact.

The Fisher never needs to be FORMED -- delta^T F delta is a scalar, equal to
Var_{v~p}(W delta). It needs p, and that is the whole question. MEASURED on the
real teacher, held out (n=320):

    teacher mass in top-k      mean     median   10th pct
       k=64                   0.9888    0.9998    0.9707
       k=256                  0.9951    0.9999    0.9877

    Fisher quadratic        1/2 dFd    err vs exact
       exact per-position     0.3890        -
       cached top-64          0.3979       2.3%
       cached top-256         0.3920       0.8%
       GLOBAL (averaged)      0.3660       5.9%

ADOPTED: per-position Fisher from a cached top-64. O(k d) = 131K flop against
508M for a full softmax -- 3,900x -- at 2.3% error, for 384 extra bytes per
position beside the int8 h_t.

Two corrections worth keeping. First, an earlier note here claimed top-k was
unsafe because "the teacher puts only 43.5% of mass in its top-64" -- that was
measured on the AVERAGE distribution, which is flat by construction. Individual
positions are extremely peaked. Averaging distributions and then reading off
concentration measures the wrong object. Second, D12 rejected exactly this cache,
but for REVERSE KL, where the STUDENT's mass escapes the teacher's support; the
Fisher is teacher-weighted, so the teacher's own top-k covers what it reads --
D12 says as much about forward KL. Same artefact, different quantity, opposite
verdict.

What remains is NOT the Fisher approximation: exact per-position Fisher recovers
63% of true KL at the starting delta and the global one 59%, so averaging costs
~4 points while the SECOND-ORDER EXPANSION costs the rest. That error shrinks
monotonically as training converges (13% at s=1.0, 5.4% at 0.5, 0.6% at 0.05),
which is the opposite of D12's top-k artefact -- a floor precisely where
resolution was needed.

Function-preserving init: the expansion is second order, so it is valid only for
small delta. That is an argument FOR the calibration step, not against the
method -- fitting the teacher->student map so delta starts small is what keeps
the objective in its valid regime.

Head alignment for a different-width teacher, measured against the real
training-time teacher (35B-A3B, d_t=2048, UNTIED output head dequantised from the
Q6_K GGUF -- no download needed, gguf.quants.dequantize, 3.3 s):

    teacher(2048) via student(2560) column space   uniform 42.4%   usage 95.2%
    student(2560) via teacher(2048)                uniform 34.8%   usage 94.2%

d_t < d_s makes the teacher->student direction rank-FEASIBLE (the 27B, at 5120,
was not), and it would leave the student's head untouched with nothing added at
inference. **It was then measured behaviourally and REJECTED.** On 321 real
teacher hidden states spanning code, prose, maths, a diff and Spanish,
reconstructing the teacher's logits through the student's head gives:

    reconstruction    KL(t||r)   KL(r||t)   top-1 agree   top-5 agree
    B_uniform           1.2106     2.0528         53.6%         56.0%
    B_usage             0.4332     0.3794         82.2%         78.1%
    -- for scale --
    5% random noise on h itself      0.0017

The best case is 250x worse than perturbing the hidden state by 5%, and loses
18% of top-1 predictions. So 95.2% of "energy explained" buys 82% agreement:
energy is nearly useless as a fidelity proxy here, exactly as D13 warned (a 4%
variance shortfall in h produced 1000%+ reverse-KL error, because the head
amplifies small h differences). Do not revive this direction on an energy number.

**ADOPTED INSTEAD: give the student the TEACHER's output head.** logits =
W_t (P h_s), P : 2560 -> 2048. Reconstruction error is then zero by construction
-- there is nothing to approximate -- and the teacher being NARROWER than the
student makes it cheap:

    factored   W_t (508.6M) + P (5.2M) = 513.8M    exact
    fused      W_t P (248320 x 2560)   = 635.7M    exact, drop-in shape
    student's current head               635.7M

Both are exact; only W_s B was lossy. Factored is 122M smaller, fused is
shape-identical to the head already there. The input embedding still needs W_s,
so it is +513.8M either way (~+13% of a 4B, ~0.25 GB in NF4).

Function-preserving init -- MEASURED, and the opposite of what was first argued.
60,000 paired (h_s, h_t) on calib_mix, fitted on the code-heavy first 80% and
evaluated on the prose tail, so the holdout is a domain shift:

    init for P                        disruption vs student   agree w/ teacher
    fit P h_s ~= h_t                  KL 0.7401  top-1 74.0%        69.7%
    fit W_t P ~= W_s, uniform tokens  KL 1.0461  top-1 66.3%        61.7%
    fit W_t P ~= W_s, usage-weighted  KL 0.0663  top-1 94.2%        82.8%
    (student today, reference)                                      83.9%

Fit P to PRESERVE THE STUDENT'S OWN LOGITS, usage-weighted -- not to match the
teacher's hidden state. Matching h_t optimises the training target at the cost of
destroying the model being started from: it costs 14 points of teacher agreement
and nearly doubles output entropy (0.91 -> 1.86 nats). The preserving init keeps
94.2% of top-1, KL 0.066, entropy 0.98, and barely moves teacher agreement.
Rank (2048 < 2560) does NOT make this direction lossy in practice; UNIFORM token
weighting does, and catastrophically.

**But the same measurements question the objective itself.** The student already
agrees with the teacher on 83.9% of top-1 predictions while its hidden state
linearly predicts h_t at only R^2 = 0.432 held out (0.624 in-domain). The two
models produce similar OUTPUTS through substantially different REPRESENTATIONS.
So distilling h is a strictly harder target than distilling logits: it demands
the student reorganise its representation, not merely reproduce behaviour. That
may be a richer signal or it may burn capacity enforcing an agreement the
student's architecture (GDN-2, compressed MLA, NoPE) has no reason to admit.
Decide it with an arm, not an argument.

Caveats on the numbers above: the usage weights were counted over all 60k
positions including the held-out tail, so 94.2% is mildly optimistic; and R^2
describes how far the target starts from the CURRENT student, not whether it is
reachable by training.

**MEASURED AT LAUNCH, AND IT KILLS THE SWAP ON THIS BUDGET.** The offline numbers
above were all computed on calib_mix, which is the corpus P's token weights were
FITTED on -- fitting and validating on the same distribution, which is the
circularity this file already warns about for PCA (D13). The first number taken
on the real training/eval distribution says something else:

    converted-untrained, NO head swap (masked150 eval@0)   ppl@8192 11.037
    converted-untrained, WITH head swap                    ppl@8192 25.858
                                                           -> 0.851 nats

Against 0.127 nats measured on calib_mix: the swap costs 6.7x more on the eval
corpus than on the corpus P was fitted against. Refitting the weights helps but
does not rescue it -- residual energy on the TRAIN distribution goes 12.30%
(calib-fitted) -> 9.22% (merged) -> 9.01% (train-fitted), so most of the 0.851
survives any reweighting.

The number that settles it: masked150's ENTIRE 150-step run moved ppl@8192 from
11.037 to 9.609, i.e. **0.139 nats total**. The head swap costs **0.851 nats at
initialisation** -- 6x the total gain a full run of this budget has ever
produced. It cannot pay for itself in 150 steps. That ratio should have been
computed before the swap was built, not after.

Three options and what each costs at init, for the record:

    A. head swap (built)              0.851 nats damage, exact target
    B. project teacher -> student     0 damage, target 0.433 nats off the teacher
    C. on-policy only, reverse KL     0 damage, exact target, no new machinery

C is the one that targets the MEASURED defect (D16: base model 0 stuck
generations, ours 28; cause `on_policy = 0.0`). A and B are cost and cleanliness
improvements to the objective, not fixes for the diagnosed problem, and either
can be evaluated later without paying a 6x handicap.

NOTE: option C still works with the llama.cpp teacher and needs no in-process
27B. The server returns h_t; the teacher's logits are then W_t h_t using the head
already dequantised from the GGUF, so reverse KL has both distributions without
the MoE loader. That part of today's work stands whichever objective is chosen.

**No MoE loader and no download are needed to do any of this.** The teacher's
head dequantises straight from the Q6_K GGUF (gguf.quants.dequantize, 3.3 s), and
llama.cpp emits h_t directly: `--embeddings --pooling none --embd-normalize -1`
returns the post-output_norm, pre-lm_head state, verified by decoding
W_t h_t into coherent next tokens (' fibonacci'->'(n', '):'->newline,
' else'->' fibonacci', max prob 0.92-0.95 on syntax). D5/D12 call the
fused-3D-expert loader the highest-value performance work; that was premised on
needing in-process teacher LOGITS, which D13's cache plus this objective remove.
The loader is still needed for ON-POLICY steps, where the teacher must see
student-generated text.


---

## Tier 2 — deployment, high confidence, orthogonal to quality

### 2.1 Quantization-aware training, weights and KV cache
**Confidence: high. Cost: medium.**

NF4 costs +4.9 to 7.1% perplexity on this model measured post-hoc. The structure
already in place is QLoRA's: a frozen base plus small trainable adapters, so
quantizing the base in the forward pass and training VeRA against it is a small
change to an established recipe.

**Add a rotation before quantizing the KV cache.** The MLA latent is the worst
case for naive quantization: CARE already removed the redundancy, so the 256
dimensions are information-dense with no slack to absorb error. A Hadamard or
learned orthogonal rotation before quantization spreads outliers across channels
and is what makes 4-bit work in QuaRot and SpinQuant. Rotation is exactly
invertible, so it is function-preserving by construction, and it composes with
the latent rotation that a positional phase would need anyway.

Caveat worth stating: this fixes nothing in Tier 1. It is required to ship, not
to make the model better.

### 2.2 Integrate the conv MTP head
**Confidence: high. Cost: low. BUILT, NOT WIRED.**

src/mtp_conv.py is a gated dilated TCN with xATLU gating, 26,627 parameters
against the built-in head's 20.45M, bit-exact identity at init, receptive field
15, predicting t+2..t+K+1 so the trunk keeps t+1. Two payoffs: multi-token
prediction is a known auxiliary objective that improves representations
(Gloeckle et al., arXiv:2404.19737), and it enables speculative decoding at
deploy without needing an engine that understands the stock MTP head.

---

## Tier 3 — cheap, plausible, quick to falsify

### 3.1 Depthwise causal conv on the MLA latent
**Confidence: medium-high. Cost: negligible.**

The MLA path is entirely pointwise (down, up_k, up_v) while the linear layers
already have a depthwise conv1d on qkv. A short causal conv over the latent c
gives the softmax layers the same cheap local mixing, which is standard in every
modern linear-attention hybrid and is where induction-like local copying comes
from. 256 x kernel = ~1K parameters per layer, 6K total; the conv state adds
kernel-1 latents to the cache.

Identity init: kernel weights [0, ..., 0, 1] pass the current position through
unchanged.

### 3.2 Learned soft head assignment ("free for all", bounded)
**Confidence: medium. Cost: negligible.**

Letting every query head attend to every kv head's keys multiplies the score
computation by n_kv and the benefit is uncertain. The cheap version is a learned
n_heads x n_kv mixing matrix on the key side, initialized to the current hard
assignment, so GQA is the starting point and the model can soften it if that
helps. 8 x 2 parameters per layer. Related: talking-heads attention (Shazeer et
al., arXiv:2003.02436) mixes across heads in the logits.

Ordering note: 1.4 captures most of the same expressiveness at zero extra FLOPs,
so do 1.4 first and treat this as the follow-up rather than the alternative.

### 3.3 Train the decay-tied positional phase
**Confidence: medium. Cost: low. VERIFIED, READY.**

lambda_t = alpha_t^(1+ic): modulus unchanged so the memory horizon is untouched,
argument c*log(alpha_t) so phase advances with accumulated forgetting. Verified
c=0 is bit-identical through chunk_kda (11.736534 and 15.962002 reproduced to
2e-07), the perturbation is mild across three decades of c (+0.49% at c=1e-2,
+3.57% at c=1e-1), and at c=1e-4 the untrained phase is very slightly better
than none. One scalar per head, reusing a cumsum the kernel already computes.

Unlike Selective RoPE (arXiv:2511.17388), which keeps rotation and decay as
independent mechanisms, the clock here is measured in nats of forgetting rather
than tokens, so there is no frequency ladder pinned to a training length. Honest
cost: positional resolution is coupled to memory horizon, so the
longest-memory channels get the least resolution.

### 3.4 Per-layer d_c allocation at fixed total cache
**Confidence: medium. Cost: low.**

All 6 MLA layers use d_c = 256 uniformly. The measured spectra differ per layer,
so allocating the same total budget unevenly should recover quality for free.
The machinery exists (allocate_ranks) and its priority function has been fixed.

Two allocation objectives now exist and they disagree. `allocate_ranks` minimises
total truncation error, which is indifferent to what a layer is FOR; it hands the
largest share (314 of 1536) to layer 3. Measured retrieval scores say layer 3
retrieves LEAST (mean 0.219) and layers 15 and 19 retrieve most (0.885, 0.826) --
see findings 0.7. `allocate_ranks_retrieval` weights the same water-filling by
retrieval score.

Granularity matters here: 8 query heads but 2 KV heads per layer means grouping
inside a layer is already per-KV-head, which is the M-LRD variant Palu
(2407.21118) reports degrades. Across layers is the axis with room.

Screening cheaply first, because d_c reshapes the latent and adapters trained at
256 cannot transfer, so a clean comparison costs one full training arm per
allocation. `src/alloc_screen.py` converts stage-AB at each allocation with NO
adapters and measures retrieval directly, which isolates the allocation's own
cost with no adapter-mismatch confound. Pre-recovery numbers rank allocations;
they do not predict trained accuracy. A separating screen buys the training arms.

---

## Drafting for decode (measured, then options)

Decode is 12 tok/s (83 ms/token) for the NF4 student through HF eager --
bitsandbytes dequantises per matmul at batch 1 and every layer pays Python
dispatch. Whatever drafts, it multiplies THAT number, so the export path
(merged adapters, absorbed MLA, an efficient engine) is worth more than any
draft mechanism until it is fixed.

### Conv MTP head -- MEASURED, not useful as a draft yet
Acceptance against the model's OWN greedy token (the quantity that makes
speculative decoding lossless), run E, 8,187 positions: head 1 4.9%, heads 2-4
~3%, chained tau = 1.06 -> 1.00x. Run D (attached, weight 0.1) was the same,
5.3%. The head is 66 K parameters with a 15-token receptive field against
Medusa's d x d per head (~6.5 M here), and it saw 1.2 M tokens. It also costs
~38% of training throughput (199 tok/s against 323) through the extra
full-vocabulary passes.
Attached, it also COST the trunk: run D 9.791 ppl@8192 against run E's 9.428
with the same head detached (Medusa-1's arrangement). Dropped from recovery.
Worth revisiting only in a dedicated phase: frozen trunk, a bigger head, many
more tokens.

### Qwen3.5's own MTP module -- available, untested here
Both checkpoints ship it (15 tensors, ~50 M in the 4B): fc(5120->2560) over
[norm(h_t); norm(emb(x_{t+1}))], one transformer layer, norm, then lm_head.
transformers ignores it on load (`_keys_to_ignore_on_load_unexpected`), so the
pipeline has been discarding a trained component.
It is SEQUENTIAL: it conditions on the true next token, so it does not ask
h_t to encode t+2 by itself. That is why DeepSeek-V3 and Qwen can train this
form attached (lambda 0.3 -> 0.1) and claim a representation benefit, and why
our parallel head could not. The teacher has the same module, so its output is
a free distillation target, exactly like the main head.
Cost as a draft: its attention layer carries a FULL-WIDTH KV cache, which
partly undoes the 4x compression while drafting -- it would want the same MLA
conversion the trunk gets.
Expectation if tried: small. The literature's gains are pretraining-scale and
Gloeckle et al. report MTP can hurt smaller models; recovery here is 1.2 M
tokens.

### Tiny diffusion drafter over the MLA latent -- idea
A small discrete-diffusion (masked-token) drafter proposes a BLOCK of k tokens
in parallel by denoising, which is what a draft wants, instead of k sequential
steps. The reason it fits this model specifically: after stage D the KV cache
IS a compact shared latent c (512 dims per layer), so a drafter can attend over
the cache the trunk already holds rather than carrying its own -- the cost that
makes the sequential MTP module awkward.
What it needs, in order: the export/decode path first (drafting a 12 tok/s
baseline is not worth engineering), then a drafter that reads c, then the
verification path (which exists: chunked continuation is exact, and the
recurrent state snapshot for rollback is ~50 MB). Unknowns: whether a small
diffusion LM proposes well enough to clear the acceptance bar that the conv
head missed, and whether block proposals survive the linear layers' recurrence
on rejection.

## Tier 4 — speculative, or expensive for the expected return

### 4.1 Mixture of VeRA experts
**Confidence: low-medium. Cost: low.**

VeRA makes adapter-MoE unusually cheap: A and B are shared and frozen, so an
expert is just the vectors d and b. Eight experts cost 8 x (rank + out_features)
per layer instead of 8 x rank x (in + out). Top-1 routing plus one always-on
shared expert (Switch plus DeepSeekMoE's shared expert) is the sensible shape.

Why it is Tier 4 despite being cheap: MoE-adapters pay off when inputs have
distinct modes, and single-task recovery distillation may not have them. At 150
steps a router is unlikely to learn a useful partition. Cheap to try, but do not
expect it to address Tier 1.

### 4.2 More tokens
**Confidence: high that it helps, but it is not an experiment.**

Current runs see 0.041 epochs of a 30M-token corpus. Everything improves with
more. Listed because it is the honest default against which every architectural
change should be compared: a change that needs 10x the steps to show a gain is
not obviously better than spending those steps on the existing model.

### 4.3 Larger recurrent state
Interference is a property of state capacity (d_k x d_v per head). Increasing it
has no teacher initialization and changes every kernel shape, so it is not a
surgery this pipeline can do. Noted so it is not rediscovered as cheap.

### 4.4 Muon or another optimizer
Kimi trained with Muon. At a 150-step budget optimizer choice plausibly matters,
but it interacts with every other result and would invalidate comparisons
against the existing arms. Park until the architecture is settled.

### 4.5 Looped reasoning (recurrent depth) -- RESEARCH FIRST, not scheduled

Added 2026-09-29 on request: to the experiments list, NOT to be started yet. The open
question is whether and how it can be added to an already-converted student.
Idea: run a block of layers several times, so the model can spend more compute per token
("latent reasoning") without growing parameters or the KV cache per token of text.

To research before any design (verify every claim against the papers themselves):
  * the retrofit literature -- converting a PRETRAINED model into a looped / recursive
    one (e.g. layer tying with small per-loop adapters) versus training loops from
    scratch; our constraint is retrofit + distillation, not pretraining.
  * which block to loop in a hybrid stack: our 24 GDN + 8 MLA layers interleave, and a
    loop must decide what happens to the GDN recurrent STATE and the MLA KV cache on
    each pass (reuse, append, or separate per iteration) -- this is the crux for us.
  * how the number of loops is set: fixed, sampled during training, or adaptive per
    token (halting / router-based), and what that costs at decode.
  * interaction with distillation: the teacher does not loop, so what does the student
    match on each pass; and with the KV-cache work (#58-#59) -- extra passes must not
    multiply the cache.
  * evaluation: GSM8K greedy + the MC suite + RULER, at matched compute and cache.
Deliverable of the research step: a short design note with the chosen variant, its
state/cache semantics, and a cheap first test -- before any code.
