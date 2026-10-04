# Adapter-MoE: routed VeRA experts

A design for adding conditional capacity between "one adapter for everything" and
a real MoE, by putting a router over a set of adapters rather than over FFN
experts. The backbone stays frozen and NF4-quantized; only vectors are learned.

## 1. Why this is much cheaper for us than in the literature

The published versions route over LoRA experts: MoLoRA / MoV (arXiv:2309.05444),
LoRAMoE (arXiv:2312.09979), MixLoRA, SiRA, MoLA (arXiv:2402.08562, "higher layers
need more LoRA experts"), HydraLoRA (NeurIPS 2024, one shared A and several B
experts). In all of them an expert is a *matrix pair*, so E experts cost
`E · r · (d_in + d_out)` parameters and top-k routing costs k separate low-rank
matmuls.

We are not using LoRA. VeRA (arXiv:2310.11454) writes the update as

    dW = diag(b) · B · diag(d) · A

where **A and B are random, frozen and shared across every layer**, and only the
vectors `d` (length r) and `b` (length d_out) are learned. So for us an expert is
*a pair of vectors*, not a pair of matrices. That changes both costs.

### 1.1 The factorization that makes routing nearly free

Write `u = A x`, the projection into the frozen rank-r basis. It does not depend
on the expert, so it is computed **once**. With per-expert `d_e` and router
weights `g_e` over the top-k experts, and `b` shared across experts:

    y = b ⊙ B ( Σ_e g_e · (d_e ⊙ u) )
      = b ⊙ B ( (Σ_e g_e d_e) ⊙ u )                      (*)

The mixture **collapses into a single effective vector** `d_eff = Σ_e g_e d_e`
before `B` is applied. So:

* `A` is applied once (as in plain VeRA),
* `B` is applied once **regardless of how many experts are active**,
* the only added work is gathering k vectors of length r and summing them.

Routing over 256 experts costs essentially the same FLOPs as routing over 2. This
is the whole point of the design, and it exists only because A and B are shared
and frozen — with LoRA experts, (*) does not factor, and k experts cost k matmuls.

Keeping `b` per-expert would break (*), because `b_e` sits *outside* `B`, forcing
k applications of B (r · d_out each). **So: per-expert `d`, shared `b`.** If
per-expert output scaling turns out to matter, the cheap version is a small
number of `b` groups (2-4), not one per expert.

### 1.2 What the expert actually is

`d_eff` is a token-dependent diagonal in a frozen random basis: the router decides
**which of the r shared directions matter for this token**. Think of it as
conditional rank reallocation rather than conditional weights.

The obvious worry is that a diagonal is too weak to be a real "expert". The
direct counter-evidence is MoV (arXiv:2309.05444), whose experts are (IA)³
*vectors* — pure diagonal scalings, strictly weaker than ours since they have no
learned basis at all — and which matched full fine-tuning with as few as 10
experts at under 1% of parameters updated. Our experts are diagonals in a
rank-1024 learned-scaled random basis, which is a strict generalisation.

## 2. Cost, for our 4B student (hidden 2560, intermediate 9216, 32 layers, r=1024)

MLP projections only (gate, up, down), one router per layer shared by all three:

| experts | expert params | routers | total | share of 4B |
|---|---|---|---|---|
| plain VeRA | 0.77 M | — | 0.77 M | 0.019% |
| 8 | 1.46 M | 0.66 M | 2.11 M | 0.053% |
| 16 | 2.24 M | 1.31 M | 3.56 M | 0.089% |
| 64 | 6.96 M | 5.24 M | 12.21 M | 0.305% |
| 256 | 25.84 M | 20.97 M | 46.81 M | 1.170% |

Note the routers dominate past E=16 — the experts are so cheap that the *gate*
becomes the expensive part, which is the opposite of every LoRA-MoE paper. Two
fixes if it matters: a low-rank router (`H → 64 → E`), or routing off the already
computed `u` instead of `x`.

Added compute: one `2·H·E` matmul per token per layer = **0.23% of the MLP's
FLOPs** at E=64. Negligible.

## 3. Load balancing without touching the objective

Standard practice is an auxiliary load-balance loss added to the training loss.
**We should not do that.** Our objective is measured in excess nats (reverse KL +
TAID + excess CE), and every term in it is interpretable in those units; adding an
unnormalised balance penalty pollutes that accounting and makes the loss curve
uncomparable with every run we have.

Use instead the **loss-free balancing** of DeepSeek-V3 (arXiv:2408.15664): keep a
per-expert bias added to the router *logits only for the top-k selection*, and
nudge it up or down from observed load. It changes routing, never the gradient,
so the loss stays exactly the objective we derived.

## 4. Where to put experts

Three signals we already have, and one worth testing:

1. **MoLA's finding** (higher layers need more experts) suggests a non-uniform
   allocation rather than E everywhere.
2. **Our retrieval-head scores** already rank layers/heads by retrieval role, and
   we already use them for grouped MLA rank allocation. The same water-filling
   can allocate experts: more where the measured role is strongest.
3. **Layer type.** The student is 24 GDN-2 linear-attention layers and 8 full
   attention. These fail differently under surgery, and expert capacity is
   probably worth more on whichever recovers worst — measurable from our
   per-layer recovery curves.
4. Start with **MLP only**. Attention `out_proj` and the MLA latents are the
   obvious next surfaces, but they interact with absorbed MLA and grouping, so
   they should not be in the first experiment.

## 5. Honest assessment of whether this will work

**The strongest reason to expect a gain:** our data mix is genuinely
heterogeneous — code across 12 languages, STEM text, long-context retrieval,
multilingual. MoE-of-adapters gains in the literature come almost entirely from
multi-task settings, because routing needs something to route *on*. We have that
structure.

**The strongest reason to doubt it:** our use is *recovery from architecture
surgery*, which may behave like a single task — approximate one fixed teacher
function everywhere. If the thing being learned is uniform across the data, a
router has nothing to specialise on and E experts converge to E copies of the same
vector. That failure is cheap to detect (expert usage entropy, and pairwise
cosine between `d_e`), and it should be checked before scaling E.

**Second risk:** the update stays rank-bounded by the frozen shared A and B. We
are adding *conditionality*, not rank. If what the model lacks is rank, this does
not supply it.

**Third risk, deployment:** plain VeRA merges into the base weights and costs
nothing at inference. A token-routed adapter cannot merge, so it stays a runtime
cost forever — small here (one gather plus a sum in rank space), but not zero, and
it interacts with the NF4 export path and QAT that come at the end.

## 6. Experiment plan

Cheap, and ordered so it fails fast:

1. **Does routing find structure at all?** Train E=8 on the existing mix, log
   expert usage entropy per layer and pairwise cosine of `{d_e}`. If entropy
   collapses or the experts become near-identical, stop — the rest is moot.
2. **A/B at fixed steps** against plain VeRA at the same rank, same objective,
   same data: ppl@8192 plus the RULER multi-key tasks. E ∈ {8, 64}.
3. **Does it route on what we think?** Log expert selection against task type
   (code vs prose vs retrieval episode) and language. Real specialisation should
   be visible as block structure in the routing matrix; if routing is
   task-independent, the gain (if any) is coming from something else and we
   should know that.
4. Only then: non-uniform allocation by retrieval score, and attention surfaces.

Step 1 costs one short run and answers the question that decides the rest.

## 7. Relation to the MoE teacher

Unrelated mechanism, easy to confuse. The MoE *teacher* (Qwen3.5-122B-A10B for
generation, 35B-A3B for logits) is a real sparse FFN MoE we consume as-is. This
design is about the *student*, which stays dense and adds conditional adaptation
on top. They can be combined without interaction.
