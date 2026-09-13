# Carmenta findings log

Running record of what we have **measured**, what the **literature** says, and what
is still **unproven**. Entries carry the run tag or citation so any claim can be
traced back. Ordered by confidence, not by date.

Last updated 2026-09-12.

---

## 1. Measured — headline results

### 1.1 Three-way validation (the presentable result)

Full pipeline vs the unmodified model it was carved out of. Both converted arms
had 150 steps / 1.23 M tokens of live-teacher recovery; the original had none.

| model | ppl@2048 | ppl@8192 | retr@4k | retr@16k |
|---|---|---|---|---|
| Qwen3.5-0.8B (original) | 12.801 | 18.285 | 13.487 | 12.689 |
| + transKDA + NoPE | 12.501 | 17.428 | 13.811 | 13.539 |
| + transKDA + NoPE + 4× KV | 12.581 | 17.440 | 13.684 | 12.823 |

Full pipeline vs original: **−1.72% ppl@2048, −4.62% ppl@8192, +1.46% retr@4k,
+1.05% retr@16k**, at 4× smaller KV cache.

**Caveat that must travel with this table:** the converted arms have each seen
1.23 M tokens of FineWeb-Edu distillation and the original has seen none, so
"better than the teacher" partly reflects that extra adaptation. The defensible
claim is *parity-or-better at 4× smaller KV after 0.047 epoch*, not *surgery
improves the model*.

The compressed arm beats the uncompressed converted arm on retrieval
(+1.05% vs +6.70% @16k). Treat as trajectory noise between two separate runs
until replicated — there is no mechanism by which compression should help.

### 1.2 Compression cost, measured against a matched control

`val-full` (d_c=256) vs `val-ctrl` (uncompressed), identical in every other
respect: NoPE, live teacher, seq 8192, 150 steps, lr 3e-5, full dense.

| | ppl@2048 | ppl@8192 | agree@8192 |
|---|---|---|---|
| control @0 | 12.345 | 17.372 | 88.65% |
| control @150 | 12.493 | 17.424 | 89.81% |
| 4× KV @0 | 13.405 | 20.362 | 81.38% |
| 4× KV @150 | 12.581 | 17.432 | 85.51% |

Compression cost **before** recovery: +8.6% @2k, **+17.2% @8k**.
**After** 1.23 M tokens: +0.70% @2k, **+0.05% @8k**.

Teacher-agreement for the compressed arm is still 4.3 pp below control →
headroom remains; neither run is converged.

### 1.3 The control got *worse* while agreeing with the teacher *more*

`val-ctrl`: a healthy uncompressed model, 294 M dense params, 150 steps
→ wikitext ppl@2048 **12.345 → 12.493 (+1.20%)** while teacher-agree@2048 went
**91.89% → 93.31%**.

Two readings, both probably true: (a) the objective is doing what it says —
pulling toward the teacher's distribution, which is not the same as minimizing
CE on held-out wikitext; (b) 294 M inherited weights are **drifting off a good
operating point** in 0.047 of an epoch. See §3.4 (pull-to-init) — this is the
failure mode that regularizer exists to prevent.

Consequence for methodology: **the arm-vs-arm gap is the meaningful quantity,
not either arm's own delta.**

---

## 2. Measured — the adaptation surface

### 2.0 On a correct objective, the dense surface IS worth it (2026-09-12)

`gate8k` vs `val-full`, live teacher, seq 8192, 150 steps, lr 3e-5, same init —
one variable, the adaptation surface:

| surface | trainable | ppl@2048 | ppl@8192 |
|---|---|---|---|
| KDA diagonal dense, rest LoRA r16 | 66.04 M | 13.405 → 13.034 (−2.77%) | 20.362 → **18.750** (−7.92%) |
| full dense KDA + attn | 293.91 M | 13.405 → **12.581** (−6.15%) | 20.362 → **17.432** (−14.39%) |

**Dense is 3.5% better @2048 and 7.0% better @8192.**

This **contradicts §2.1**, which found the same three surfaces nearly tied on the
*cached* teacher (1.2% spread). The gap on the live teacher is ~10× larger.
Working hypothesis: the cached top-64 objective was too weak and too misdirected
(gradient 48° off-target, §4.1) for any surface to exploit extra capacity — so
an ablation run on a broken objective measured nothing about capacity at all.
**Not isolated**; stated as a hypothesis.

Remaining confound: both arms ran at **3e-5**, the *dense*-safe rate. The 66 M
surface may prefer a higher LR, and that arm has not been run. It is the one
result that could still narrow this.

Methodological lesson worth keeping: **an ablation is only as trustworthy as the
objective it runs on.** §2.1 was measured carefully and was still worthless.

### 2.0b LayerScale at identity init buys nothing (2026-09-12)

`ls8k` = `val-full` + per-channel `(1+λ)` on every residual branch output,
λ=0 (identity), fp32, installed after LoRA.

| step | `val-full` | `ls8k` |
|---|---|---|
| 50 | 12.813 / 17.695 | 12.817 / 17.693 |
| 100 | 12.498 / 17.292 | 12.498 / 17.302 |
| 150 | **12.581 / 17.432** | **12.590 / 17.463** |

A dead heat, very slightly worse. Consistent with §3.3: λ is redundant wherever
the branch output is trained densely, and the only LoRA-only branch (the FFN) is
too small a lever to move a 294 M-param run.

**Install bug, disclosed:** the wrap matched target names by SUBSTRING, so a
LoRA-wrapped module was wrapped at both `<target>` and `<target>.base` — 96
wraps where 48 were intended, 98,304 params not 49,152. Both copies are identity
at init so bit-exactness held, and both were trainable, meaning the arm had
*twice* the intended capacity and still tied. The negative result stands and is
if anything stronger. Fixed with an `endswith` match.

**Both arms peak at step 100 and regress by 150** (val-full 12.498 → 12.581).
More drift evidence; see §3.4 (pull-to-init).

### 2.0c Rank 16 cannot represent what the dense surface does (2026-09-13)

Biderman's test on our own `val-full` weights: SVD of `W_trained − W_base` for
the KDA projections that run trained **densely**, which is exactly where the
LoRA arms used rank 16.

| matrix | rank@50% var | rank@90% var | var kept by top-16 | by top-32 | full |
|---|---|---|---|---|---|
| `in_proj_qkv` | **303** | 794 | **6.2%** | 10.0% | 1024 |
| `out_proj` | **183** | 620 | **11.1%** | 17.1% | 1024 |

Rank 16 captures **6%** of the dense update on `in_proj_qkv`. Biderman reports
full-FT perturbations at 10–100× typical LoRA rank; ours are 20–50×.

**Caveat:** this shows rank 16 cannot reproduce the *dense solution*, not that no
low-rank solution reaches the same loss — a dense optimizer takes a high-rank
path partly because nothing stops it. Suggestive, not conclusive.

Regime note: our objective (perplexity at 8192 after surgery) is **CPT-shaped,
not IFT-shaped**. Biderman: LoRA matches full FT on instruction finetuning at
r=256, but "in CPT, LoRA underperforms full finetuning across all
configurations." LoLCATs says the same in its own limitations section for
linearized models (up to 42.4 points on 5-shot MMLU).

### 2.0c-ter Per-OPERATOR rank: in_proj_a is low-rank, the rest are not (2026-09-13)

`in_proj_qkv` is a FUSED parameter — q|k|v stacked — so measuring its rank as
one matrix conflates three operators whose updates live in different row spaces.
Unfusing (user's catch):

| operator | rel ‖ΔW‖/‖W‖ | rank@50% | top-16 var | params |
|---|---|---|---|---|
| **`in_proj_a`** | 0.0026 | **26** | **46.1%** | 37.75 M |
| `in_proj_qkv` (fused) | 0.0038 | 303 | 6.2% | 113.25 M |
| — q only | | 219–235 | 6–9% | |
| — k only | | 204–238 | 6–12% | |
| — v only | | 202–237 | 5–12% | |
| `in_proj_z` | 0.0045 | 218 | 7.5% | 37.75 M |
| `out_proj` | **0.0106** | 183 | 11.1% | 37.75 M |

**Fusion inflation is real but modest**: 289 fused vs ~210 per operator, ~1.4x —
not 3x. (The individual ranks sum to ~625 while the fused matrix is 289, so q, k
and v updates share most of their input subspace.) The dense-for-KDA conclusion
survives for qkv/z/out_proj.

**The find: `in_proj_a` is intrinsically low-rank.** Mechanistically obvious in
hindsight — Stage B built it by `repeat_interleave` of a (16,1024) matrix, so it
STARTS at exactly rank 16, and gradients across the tiled channels within a head
stay highly correlated. Its update is 7–12x lower rank than any other KDA
operator, at a comparable relative magnitude (so not noise).

**Consequence: `gate8k` (§2.0) had the split exactly inverted.** It made
`in_proj_a` dense (37.75 M, where LoRA r16 captures 46%) and gave
`in_proj_qkv`/`in_proj_z`/`out_proj` LoRA r16 (capturing 6–11% of *their*
updates). The dense budget went to the one low-rank operator and starved the
three high-rank ones. That is a mechanical explanation for §2.0's 7% gap that
does NOT require "LoRA lacks capacity" — it requires only that LoRA was pointed
at the wrong matrices. **§2.0's conclusion ("the dense surface is worth it") is
therefore not established.**

Implied configuration, by measurement rather than by which parameter was tiled:
`in_proj_a` → LoRA r32; `in_proj_qkv`, `in_proj_z`, `out_proj` → dense; FFN →
LoRA. Saves 37.75 M of 294 M (13%). `out_proj` has the largest relative update
(0.0106) at the lowest rank of the three, so it is the highest-value dense
parameter per unit of capacity.

Methods note: measure per-operator, never on a fused projection.

### 2.0c-bis Per-head adapters do NOT beat a global one (2026-09-13)

KDA heads are independent in the recurrence, and `in_proj_qkv` (6144,1024) is
q|k|v stacked, each 16 heads x 128 — so 48 blocks of (128,1024). Hypothesis: a
global rank-r adapter forces all 48 blocks to share ONE r-dim input subspace,
while per-head adapters give each its own.

**The structure is real.** Mean pairwise overlap of the heads' top-16 input
subspaces is **0.070** (1.0 = identical, 0.0 = orthogonal). Per-head, each block
needs rank 33–36 of 128 for half its energy, against 289 of 1024 globally.

**But it buys nothing at matched parameter budget:**

| budget | global rank | var | per-head rank | var |
|---|---|---|---|---|
| 0.11 M | 15 | 9.6% | 1 | 5.4% |
| 0.50 M | 69 | 20.0% | 9 | 20.1% |
| 2.00 M | 279 | 48.9% | 36 | 49.6% |
| 3.00 M | 418 | 63.3% | 54 | 64.2% |

Within a percentage point either way; global wins at small budgets. Reason: the
head subspaces being near-orthogonal means the **global SVD already discovers
the block structure on its own** and allocates separate directions to separate
heads. Imposing it as a constraint adds no information.

**Conclusion: the update is irreducibly high-rank at every structural level.**
Capturing 50% costs 2.0 M of 6.29 M dense params (32%); 63% costs 3.0 M (48%).
No rank-structured adapter gets a useful fraction of this update cheaply — a
stronger statement than "rank 16 is too small". Dense is close to optimal for
KDA; the L/XL scaling lever has to be something else (a staged objective that
requires less movement, per LoLCATs, or fewer dense params by construction).

**Caveat:** these runs drift (§1.3, §2.0b), and a drifting dense run accumulates
high-rank *noise* on top of the useful update, which would inflate these ranks.
Re-measuring under a pull-to-init penalty (§3.4) would separate the two.

Methods: `svdvals` on the (128,1024) blocks returns NaN in fp32 on this box —
use fp64. Silent, and it produced a full table of NaN before being caught.

### 2.0d BUG: every run before 2026-09-13 had TWO LoRA adapters per target

`train_recovery.py` calls `inject_lora` twice when `--init-adapters` is set --
once to shape the model for the load, once for real. `inject_lora` matched rule
patterns by **substring**, and a wrapped target exposes its frozen base at
`<target>.base`, which contains the pattern. So the second pass wrapped the base
of every already-wrapped module:

    out = base(x) + lora_inner(x) + lora_outer(x)

Two parallel rank-16 adapters -- **effectively rank 32 at 2x the adapter
params** -- in `val-full`, `val-ctrl`, `gate8k`, `ls8k` and every earlier run.
Confirmed in the saved keys: `in_proj_qkv.base.base.weight` alongside both
`in_proj_qkv.lora_A` and `in_proj_qkv.base.lora_A`.

Direction of the error: the LoRA arms had **more** capacity than reported, so it
weakens the capacity explanation for §2.0's gap and strengthens the learning-rate
one. Fixed with an `endswith` match (same bug class as §2.0b's LayerScale
install -- substring matching against a wrapper's inner `.base`).

`gate8kv2` / `gate8kv2-lr` re-run the surface pair on corrected code.

### 2.1 Three surfaces, matched budget (SUPERSEDED by §2.0 — broken objective)

Same start (13.40 / 20.33), 150 steps, seq 2048, **cached top-64 teacher**:

| surface | trainable | lr | ppl@2048 | ppl@8192 | run tag |
|---|---|---|---|---|---|
| LoRA r16 + A_log/dt_bias | 25.7 M | 2e-4 | 12.70 | 17.71 | `mla-care-d256-r2` |
| + `in_proj_a` dense | 63.5 M | 2e-4 | 12.69 | 17.68 | `gate-r4` |
| full dense KDA + attn | 293.9 M | 3e-5 | 12.68 | **17.50** | `attn-lr3e5` |

11.4× the parameters buys 0.02 ppl @2k and 0.21 ppl @8k. The two small arms
differ from each other by 0.03, so the dense advantage is ~7× that gap.
**Suggestive, not established.** Three confounds:

1. All three ran on the **cached top-64 teacher** — the defective objective
   (§4.1), which corrupts precisely the long-range behaviour extra capacity
   should buy.
2. **LR is confounded with surface**: 2e-4 for the small arms, 3e-5 for dense,
   because dense at 2e-4 blows up (§2.3).
3. 150 steps = **0.012 epoch**. Capacity shows up with data; early on a bigger
   surface is strictly worse (more to move, lower LR).

A clean re-run on the live teacher at seq 8192 is in progress (`gate8k`).

### 2.2 Throughput is *not* the cost of a dense surface

`gate-r4` (63.5 M) 1222 tok/s vs `attn-lr3e5` (293.9 M) 1210 tok/s at seq 2048 —
**~1%**. The backward already flows through those layers to reach the LoRA
gradients; dense only adds the weight-gradient matmul.

The real costs of the dense surface are: optimizer state, the 1.17 GB resume
file (vs ~300 MB), and — most importantly — it **drags the whole model onto a
dense-safe 3e-5** because there is a single parameter group. The FFN LoRA is
trained ~6.7× slower than it wants to be.

### 2.3 Learning rate does not transfer between surfaces

`attn-r4`: 293.9 M dense at **lr 2e-4** → ppl@2048 13.40 → **24.79 (+85.0%)**,
ppl@8192 20.33 → **32.43 (+59.5%)**. (Earlier measurement of the same effect:
163% worse before annealing back to 104%.) **3e-5 is the dense-safe value.**

### 2.4 KDA parameter breakdown (18 layers, post stage A+B)

| submodule | params | share | layer-0 shape |
|---|---|---|---|
| `in_proj_qkv.weight` | 113.25 M | 49.4% | (6144, 1024) |
| `in_proj_a.weight` | 37.75 M | 16.5% | (2048, 1024) |
| `in_proj_z.weight` | 37.75 M | 16.5% | (2048, 1024) |
| `out_proj.weight` | 37.75 M | 16.5% | (1024, 2048) |
| `a_lora_B` / `a_lora_A` | 1.77 M | 0.8% | (2048, 32) / (32, 1024) |
| `conv1d.weight` | 0.44 M | 0.2% | **(6144, 1, 4)** |
| `in_proj_b.weight` | 0.29 M | 0.1% | **(16, 1024)** |
| `A_log` + `dt_bias` | 0.08 M | 0.04% | (16, 128) each |
| **total** | **229.08 M** | | |

**The tiled diagonal is 0.08 M — 0.04% of KDA.** `A_log` and `dt_bias` are
unconditionally in `also_train`, so *every* run in this project has trained
them, including the ones labelled "LoRA only". The tiled parameter has never
been the variable in any comparison.

**LoRA is degenerate on two of these**, forced by shape, not by preference:
- `in_proj_b` is (16, 1024) — rank 16 **is** full rank, and the adapter would
  cost more than the weight.
- `conv1d` is (6144, 1, 4) — depthwise, no cross-channel mixing to factorize;
  the only axis with extent is the kernel, length 4, so full rank is 4.

Both are therefore trained **dense**; 0.73 M combined.

### 2.5 `in_proj_z` had no adaptation path at all (found 2026-09-12)

37.75 M, 16.5% of KDA — no LoRA rule, frozen in every LoRA-only arm, dense only
under `--train-attn`. An oversight, not a design choice. Fixed: r16 rule added.
Dense runs now also carry 0.88 M of redundant adapter. **`val-full`'s recorded
numbers predate this rule.**

---

## 3. Literature

### 3.1 LayerScale — verified, and it does not support the original plan

**CaiT** (Touvron et al., [arXiv:2103.17239](https://arxiv.org/abs/2103.17239)):
learnable per-channel diagonal on each residual **branch output**.

- Init is **depth-dependent and not small**: ε = **0.1** up to depth 18, 1e-5 at
  24, 1e-6 deeper. Near-zero init is a *very deep network* phenomenon.
  *(This corrects an earlier claim in this project that the benefit requires
  ~1e-4.)*
- At depth ≤18 LayerScale is a **wash** against a single learnable scalar
  (80.5 vs 80.4 @12; 81.7 vs 81.6 @18). The per-channel part earns ~1.3 pp only
  at depth 24–36, ImageNet from scratch.
- **No frontier text LLM uses it.** Verified by grepping the installed
  `transformers` source: Llama, Mistral, Gemma, Qwen2/3/3-Next all have bare
  `residual + hidden_states`. All 34 LayerScale-family hits are vision, audio or
  detection modules. (Lone counterexample: Zyphra ZAYA's `ZayaResidualScaling` —
  and it is **identity**-initialized.)

### 3.2 Zero-init gates work on ADDITIVE branches only

Flamingo `tanh(α)` ([2204.14198](https://arxiv.org/abs/2204.14198), removing it
costs −4.2% + instability), LLaMA-Adapter ([2303.16199](https://arxiv.org/abs/2303.16199),
40.77% → 83.85%), ControlNet zero-convs ([2302.05543](https://arxiv.org/abs/2302.05543)),
LoRA's zero `B`, ReZero ([2003.04887](https://arxiv.org/abs/2003.04887)),
SkipInit, Fixup.

**Shared principle:** the gate sits on a branch that is purely *additive* to an
otherwise-intact pretrained network, so gate=0 reproduces the pretrained model
exactly. The benefit is "don't let a random new module corrupt a good
representation early" — **not** optimization conditioning.

**This does not transfer to our grafts.** KDA and MLA are **replacements**.
Zeroing them does not give the teacher; it gives the teacher with its token
mixer deleted, and throws away the surgical init (GDN tiling, whitened SVD) we
engineered. **MOHAWK** ([2408.10189](https://arxiv.org/abs/2408.10189))
initializes its Mamba-2 gate **open, at 1**, explicitly to cancel the gate.

**No attention→linear-attention paper uses a zero-init gate.** The field's
answer to "blend the new module in gradually" is staged output-matching losses
and progressive layer replacement:
- LoLCATs ([2410.10254](https://arxiv.org/abs/2410.10254)): attention-output MSE, then LoRA
- MOHAWK: matrix orientation → hidden-state alignment → distillation
- Mamba-in-the-Llama ([2408.15237](https://arxiv.org/abs/2408.15237)): progressive replacement
- Liger ([2503.01496](https://arxiv.org/abs/2503.01496)): builds the gate *from* pretrained key weights
- MHA2MLA ([2502.14837](https://arxiv.org/abs/2502.14837)): our exact MLA surgery, 0.3–0.6% of pretraining data, no gate

### 3.3 The surviving argument for a per-channel scale is DoRA's

`diag(1+λ)W − W = diag(λ)W` is generally **full rank**, so a rank-*r* adapter on
that module cannot express it. That is the magnitude component of **DoRA**
(Liu et al., [2402.09353](https://arxiv.org/abs/2402.09353)): +3.7% over LoRA on
LLaMA-7B commonsense (78.4 vs 74.7), +1.0% on 13B. Related identity-initialized
per-channel adapters: **(IA)³** ([2205.05638](https://arxiv.org/abs/2205.05638)),
**SSF** ([2210.08823](https://arxiv.org/abs/2210.08823), +11.48 pp over full FT on VTAB-1k).
**All three initialize at identity.**

Where it is redundant for us: the KDA branch (`linear_attn.norm` is already a
trainable per-channel gain before `out_proj`, separated by *nothing but a linear
map*, plus `z` gives a data-dependent SiLU gate), and any module trained dense
(dense already reaches any row scaling). **Non-redundant only where the branch
output is LoRA-only — i.e. the FFN.**

### 3.4 Depthwise-conv PEFT: don't factorize, and pull-to-init

Literature exists (LoCA, ECCV 2026 arXiv:2607.06918; LoRA-Edge, DATE 2026
arXiv:2511.03765; CoLoRA arXiv:2505.18315; edge-PEFT arXiv:2507.23536). What is
unwritten is our exact case: a `(C, 1, K)` **1-D causal** kernel at `K=4`.

- **LoCA's own parameter accounting inverts at K=4**: `P_LoCA ≈ r·C +
  const(k⁴+k²)` vs `P_LoRA ≈ r·k²·C`, motivated for *"large-kernel operators,
  such as the 7×7 depthwise convolution"* (k²=49). At k=4 the k⁴ term swamps the
  saving.
- **The depthwise conv is not where adaptation capacity belongs.** LoCA's
  MobileMamba ablation reproduces MambaPEFT's finding: *"Adapting the mixer
  yields the largest performance improvement, whereas adapting only dw captures
  mainly local information and exhibits performance variance."* MobileMamba is
  an SSM hybrid — nearest neighbour to our setting.
- **Actionable:** MambaPEFT ([2411.03855](https://arxiv.org/abs/2411.03855))
  replaces `|W|²` weight decay with **`|W − W_pretrain|²` at ~1e-3** to keep an
  unfrozen pretrained operator from drifting. We run `weight_decay=0.0`. This
  targets exactly the drift measured in §1.3, and applies to the whole dense
  surface, not just the conv. **Untested here — queue as an arm.**

Skeptical flags the researcher raised (keep these attached):
LoCA's appendix pseudocode does `out_c, in_c, kh, kw = conv.weight.shape` then a
diagonal extraction — for a depthwise kernel `in_c == 1`, so that diagonal has
length 1; unresolved without reading the repo. LoRA-C's released code is
verbatim Microsoft `loralib.ConvLoRA` and contradicts its own axis narrative —
treat as unreliable. FSF (ICLR 2025) had no verifiable arXiv ID; cite by
OpenReview or not at all.

---

## 4. Traps — things that silently destroy the model

### 4.1 The cached top-64 teacher was defective in two independent ways

**Truncation.** At k=64 over a 248,320 vocabulary, top-k distillation closes
~**5%** of the gap between plain CE and full distillation, with a parameter
gradient **~48° off** the full-distillation direction at 1.8× the norm. The true
next token falls outside the cached top-64 on **10.8%** of positions, where the
loss says nothing.

**Context.** A cache built on 2048-token blocks conditions each stored
distribution on ≤2047 tokens while training draws windows up to 32768. Forward
KL is mode-covering, so a teacher that cannot see the needle spreads its mass and
the loss **penalizes** a student that retrieves correctly. At a 4096-token gap:
teacher NLL on the needle **2.01 → 16.50**, top-1 **87.5% → 0%**, and a
retrieving student scored **+2.90 nats worse** than one mimicking the uninformed
teacher.

Live teacher removes both at ~30–50% wall-clock and gives every layer's hidden
states for free. **Caches wiped 2026-09-12.**

### 4.2 Stage A: Qwen3.5's RMSNorm is zero-centered

`out = x/rms(x) · (1 + weight)`, weight init **0**. Fold `(1 + weight)` and reset
to `0.0`, not `1.0`. Folding `weight` directly → max logit delta **25.1**, top-1
agreement **3.9%**.

Three norms deliberately not fused: final `model.norm` (tied to `lm_head`),
`q_norm`/`k_norm` (gain applied *after* normalization — a nonlinearity sits
between projection and gain), `linear_attn.norm` (opposite ±1 convention).

### 4.3 bf16 norm gains receive exactly zero update

At `w ≈ 0.5` the bf16 ULP is **3.91e-3** while an Adam step at 2e-4 is ~2e-4 —
the update **rounds to exactly zero**. `q_norm`/`k_norm` got no gradient at all;
`linear_attn.norm` and `model.norm` undershot by 48–86×. All 55,552 gains are
cast to **fp32** before optimizer construction. Same reasoning applies to any new
per-channel parameter (LayerScale λ included).

### 4.4 Checkpoint on `requires_grad`, never a name list

A hardcoded name filter silently discarded `in_proj_a`'s **37.7 M** dense
params — the run trained them, the eval curve showed the benefit, the weights
were dropped on save.

### 4.5 Other silent failures

- **Stale length-mix cap** fired for `--live-teacher`, truncating the mix to
  8192-only.
- **`allocate_ranks` was never called, and its priority function was backwards**
  (`S2[r]/tail` instead of `S2[r]`) — gave 67% of the budget to the layer that
  compressed *best*.
- **OneCycleLR bakes `total_steps` into `state_dict`** — resume with a mismatched
  `--steps` raises `ValueError: Tried to step 9 times`. Now guarded explicitly.
- **`ps -eo args | grep "[t]rain_recovery.py"` self-matches the shell.** Match on
  `comm == python`. A too-narrow process check (capped output) caused a
  GPU collision on 2026-09-12 that OOMed a launch.
- **Never `pkill -f`.** Kill by explicit numeric PID.

---

## 5. Infrastructure — measured

- **Gradient checkpointing: 1.43×.** At 8192 the un-checkpointed peak is
  25.5 GiB (fits); at 32768 it does not, so checkpointing is mandatory there.
  Length-conditional via `--ckpt-above`.
- **`causal_conv1d` built for sm_87** (`TORCH_CUDA_ARCH_LIST=8.7`, source build —
  no published wheel targets this board): **12×** on that op, correct to 5.5e-3
  relative, runs 18× per forward.
- **No `torch.compile`.** Inductor decomposes SDPA instead of preserving
  FlashAttention, materializing a **16 GiB** n² attention matrix at 32k. 13%
  regression at short context (0.87×), OOM at long.
- **Resume verified end-to-end**: killed at step 8/12, resumed at the saved step,
  curve continuous (ppl@8192 20.362 → 17.256 → 17.295), 0 errors; mismatched
  `--steps` refused loudly. ~1.1 GiB per write (dense surface).
- **Throughput**: ~433 tok/s at seq 8192 with live teacher; ~1220 tok/s at
  seq 2048 with cached teacher. 150 steps at 8192 ≈ 48–50 min.
- **Dispatch-bound**: 82% CPU dispatch at short context.

---

## 6. Unproven / open

- **CARE rank allocation.** Water-filling implemented and corrected. Whitened
  version: **−5.02%** summed activation error at identical 4.00× KV, and
  starting perplexity **0.8% worse**. Optimizing its own proxy is not evidence.
  **Off.** Current allocation over budget 1536:
  `{3: 314, 7: 203, 11: 204, 15: 280, 19: 285, 23: 250}`.
- **SSMax.** Premise confirmed on this model: mean attention entropy climbs
  **+0.98** from 512→4096, per-head scaling cuts that climb by **54%**.
  48 params, bit-exact no-op at the reference length in bf16.
  **Implemented, never trained.**
- **Decay seeding (transKDA init).** 13-arm ablation: channel assignment is
  high-variance and unpredictable — control **σ = 1.26** on ppl@32768, best
  single arm a **random permutation**, weight-derived beat random by 0.3σ. Not
  evidence. Tiled init is bit-exact and is the default.
- **MLA blend gate.** Implemented (`stage_d_transmla.py`, `s` init −6.0 →
  sigmoid 0.0025), `blend=False` by default, **never passed by
  `train_recovery.py`** — no run has used it. Analysis says a *learned* gate
  would stay shut (the loss prefers the uncompressed path it interpolates
  against); an *annealed* schedule is a different and possibly viable mechanism.
- **Vision.** ViT intact and untouched at the original checkpoint — 153 tensors,
  its own 48×48 absolute position grid, internal 2D rotary, none reachable by
  language-trunk surgery. Dropped at load, not damaged. Reattachment unwritten.
- **Attention distillation.** `train_transfer.py` exists (LoLCATs-style
  layer-local hidden-state MSE) and is **invoked by no recovery run**. Matching
  attention maps is O(n²) — 16 GiB per layer at 32k.
- **Convergence.** No run in this project has reached it. Every one ended on
  wall-clock with the loss still descending.

---

## 7. Retracted — claims this project made and then disproved

Kept deliberately: every one was stated confidently before it was checked.

| claimed | actual |
|---|---|
| "+1.18%, 93% recovered" | +4.8% |
| "+2.93% improvement" | noise |
| "torch.compile will help" | 0.87× regression, then OOM |
| "long windows are near-free" | 14% throughput loss |
| "MLP dominates activations" | KDA does |
| "the 0.8B has no vision tower" | it does — 153 tensors |
| "LayerScale's benefit needs ~1e-4 init" | CaiT uses 0.1 up to depth 18 |
| "λ after LoRA vs before is what makes it work" | both are trainable; ordering is a minor modelling choice |
| "identical init prevents diversification" | measured — it diversified 1.8% |
| "KDA uses scalar decay" | it uses a diagonal |
| "val-ctrl died at step 100" | it was alive; the process check was too narrow |
| "retrieval cost −17.32% / −30.39%" | mostly training damage from the broken objective |
