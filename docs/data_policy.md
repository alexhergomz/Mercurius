# Data policy

Everything the student trains on must be openly available, licensed for
commercial use without share-alike or non-commercial terms, and must not
infringe anyone's rights or privacy. Where a source cannot be shown to meet
this, it is left out. The same filters are applied mechanically, by script,
so the policy is enforced rather than remembered.

## 1. Licenses

| kind | admitted | excluded |
|---|---|---|
| code | MIT, MIT-0, Apache-2.0, BSD-2/3-Clause, BSD-3-Clause-Clear, 0BSD, ISC, Unlicense, CC0-1.0, Zlib, PSF-2.0, BSL-1.0, UPL-1.0 | no license, NOASSERTION (until the license text is classified), GPL/LGPL/AGPL, MPL, EPL, SSPL, BUSL, any non-commercial term |
| data / text | CC-BY-4.0, CC0, ODC-BY, Apache-2.0, MIT | CC-BY-SA (share-alike), CC-BY-NC*, "no license tag", custom model licenses with use restrictions |

The license is read from the **source**, not from a dataset's own tag: a
CC-BY dataset of scraped code carries each repository's license, not CC-BY.
For repositories this is GitHub's detected SPDX id (`scripts/repo_licenses.py`).

Decision (user, 2026-09-22): CC-BY-SA and NOASSERTION stay excluded for now; the
NOASSERTION repositories can be revisited by classifying their license text.

## 2. Who generated it

Model-generated text (reasoning traces, agent trajectories) is admitted only if
the generating model's license permits using its outputs to train other models:
Qwen (Apache-2.0 releases), DeepSeek (MIT), GLM (MIT), gpt-oss (Apache-2.0),
NVIDIA Nemotron outputs released under CC-BY-4.0. Outputs of models whose terms
forbid training competing models are excluded -- Claude, GPT-4-class API
models, Gemini -- even when the dataset itself carries an open license. This
removes, for example, SWE-smith's trajectories and allenai/tmax-sft-big.
Datasets that do not state the generator are excluded until they do.

## 3. Rights and privacy

- Repository **code, tests and docs** are covered by the repository's license.
  GitHub **issue and PR text is not**: it belongs to its authors and is not
  licensed under the repository's terms. Self-built episodes therefore take
  their tasks from the code itself (tests, docstrings, commit-free synthetic
  tasks written by the teacher), not from scraped issues.
- Secrets and personal data are scrubbed from every document before use:
  e-mail addresses, API keys and tokens, private keys, and author/contact
  headers.
- The Stack's opt-out list ("Am I in The Stack") is honoured for any
  repository-derived content.

## 4. Contamination

Excluded from training regardless of license: the twelve SWE-bench (Verified)
repositories, and any repository or problem set that overlaps an evaluation we
report (HumanEval/MBPP, GSM8K/MATH test splits, LiveCodeBench, RULER).
Verified on the current candidate list: none of the twelve SWE-bench
repositories occurs in SWE-rebench-V2, R2E-Gym, SWE-smith or SWE-Gym.

## 5. Format: train on what the model will actually see

Agent data is not flattened. A repository episode is rendered through the
Qwen3.5 chat template as the deployed agent would see it: a system prompt with
the real tool schemas, then tool calls (`list_dir`, `read_file`, `grep`,
`search_code` -- retrieval over the repository's own chunks, `run_tests` where
an environment exists) and tool results containing the actual repository
content. Assistant turns are short and come from the teacher; the long tool
results are real repository text, which costs prefill only.

## Current candidate repositories

`data/repos/licenses.json`: 3,790 repositories behind the open SWE task sets.
3,187 admitted (MIT 1,600, Apache-2.0 1,253, BSD-3 222, other permissive 109),
covering 74,866 tasks. Excluded: 501 NOASSERTION, 77 without a license, 12
copyleft, 4 not found.

## Generation economics (for the FULL build, not the pilot)

Episode throughput is set by the teacher's DECODE speed, not by the
repositories: the 27B at 4-bit decodes ~8 tok/s per slot (memory-bound, ~17.6
GB of weights against ~273 GB/s), while prefill runs at ~520 tok/s and tool
results are prefill only. Measured on the pilot: ~0.4 M episode tokens per
hour, median episode 9.0 k tokens over 7 assistant turns -- roughly 1.2 k
tokens of real repository content per generated decision.

So the lever for the full build is tokens of context bought per generated
token, not more workers (slots share the same memory bandwidth):

  * raise repo_env.MAX_READ_LINES (400) and the default search_code k (5), so
    each tool result carries more real content;
  * raise the rollout turn cap (20) for repositories large enough to sustain
    exploration;
  * prefer larger repositories, whose files and search results are longer.

Left at pilot settings deliberately: the pilot measures quality and filter
rates, and changing the economics at the same time would confound both.


## 6. Task statements must come from code, not from people

SWE-style trajectory datasets prompt the agent with a problem statement. Where
that statement is a real GitHub issue or pull-request description, it is prose
written by a person and is NOT covered by the repository's licence, whatever
licence the dataset collection carries. Such rows are rejected:

| source | statement | verdict |
|---|---|---|
| AweAI-Team/Scale-SWE | card: "the issue description conveying the bug"; no licence tag | rejected |
| nebius/SWE-rebench-V2 | card: "derived from real GitHub issues and pull requests" | rejected |
| nvidia/SWE-Zero, SWE-Hero | prompts are verbatim bug reports | rejected |
| R2E-Gym (SWE-GEN) | synthesised from the commit by back-translation (arXiv:2504.07164) | accepted |
| SWE-smith | tasks synthesised by perturbing code | accepted |
| our own episodes | written by the teacher from the repository's code | accepted |

Consequence, measured: nvidia/Open-SWE-Traces is excluded ENTIRELY -- every row
sampled traces back to Scale-SWE or SWE-rebench -- even though it was the
strongest candidate on every other axis. nvidia/Nemotron-SWE-v1 (R2E-Gym tasks,
Qwen3-Coder trajectories) passes. Unknown provenance counts as rejected, not as
clean. A regex over the first messages also drops anything that still reads like
a pasted human report (bug-report templates, issue URLs, "@user wrote").

Licences stated as a URL rather than an SPDX id are resolved against GitHub's
own detection before judging them, so a permissive repository is not dropped
over a formatting difference; NOASSERTION remains excluded.

## 7. Attribution

scripts/build_attribution.py writes data/ATTRIBUTION.md (every repository with
licence, commit and token count; every third-party dataset with licence,
task provenance and generator) and data/NOTICE, the short form to ship with a
model release. Permissive does not mean obligation-free: MIT, BSD and Apache
require the notice to travel with substantial portions, and CC-BY requires
attribution. Copyright headers inside repository content are deliberately NOT
scrubbed for this reason (only secrets and e-mail addresses are).

## 8. Ethical review

What is checked, and what it costs:

- **Consent.** The Stack's opt-out requests are honoured, including the
  requester's own account even when they listed only some repositories: 8,219
  requests, 14 of our admitted repositories removed.
- **Privacy.** Secrets and e-mail addresses are replaced in every message, not
  only in tool output. Names in licence headers are kept, because attribution
  requires them -- a deliberate trade recorded here rather than silently made.
- **Authorship.** No issue or PR prose (section 6); no scraped forum or chat
  text; no model outputs whose terms forbid training.
- **Evaluation integrity.** SWE-bench Verified repositories excluded from
  training; RULER and WikiText held out.
- **Not currently checked, and worth stating:** repository content is not
  screened for offensive comments or for dual-use security tooling. Both exist
  in open-source code. A content screen over tool outputs would be cheap to add
  if wanted; today the only content filters are the secret/e-mail scrub and the
  binary/vendored exclusions.

## 9. Audits actually run (2026-09-23)

| check | result |
|---|---|
| licence TEXT vs GitHub's SPDX tag, 144 clones | 0 repositories carry added restrictions (Commons Clause, non-commercial, no-derivatives, "may not be used to train"). Two Apache-2.0 files flagged by a first pass were false positives -- standard Apache text -- and the rule was corrected rather than the finding waved away |
| AI opt-out markers (`.noai`, `NOAI`, `ai.txt`, robots directives) in clones | 0 found; the check stays in case later batches have them |
| The Stack opt-out list | 8,219 requests parsed; 14 admitted repositories removed, including the requester's own account even when they listed only some repositories |
| task-statement provenance | Open-SWE-Traces rejected entirely (every sampled row traces to Scale-SWE or SWE-rebench, i.e. human issue text); Nemotron-SWE-v1 (R2E-Gym, synthesised statements) accepted |
| model licences | Qwen3.5-4B, Qwen3.5-27B and the 4-bit GGUF are all Apache-2.0 and ungated, so outputs may be used as training targets and a modified model may be released with a NOTICE and a statement of changes |
| corpora | training data is ODC-BY (FineWeb-Edu and things derived from it). WikiText (CC-BY-SA/GFDL), PG-19 and the Paul Graham essay haystack are EVALUATION ONLY and never trained on -- the essays in particular are the author's copyright whatever the mirror is tagged |

Residual items, stated rather than hidden: the PG-19 mirror carries no licence
tag (the books themselves are public domain); the RULER haystack should be
swapped for public-domain text before any evaluation artifact is distributed;
and repository content is not screened for offensive comments or dual-use
security tooling.

## 10. Third-party trajectory sources: what was rejected, and why (audit, 2026-09-23)

Search of the public agent-trajectory sets that match our format:

| dataset | licence | verdict |
|---|---|---|
| nvidia/Nemotron-SWE-v1 (r2e_gym) | CC-BY-4.0, per-row repo licence | **accepted**, capped at 1 M tokens per repository |
| nvidia/Open-SWE-Traces | CC-BY-4.0 | rejected: prompts are GitHub issue text |
| nvidia/Nemotron-SFT-SWE-v2 / v3.5 | CC-BY-4.0 | rejected for now: prompt pools mix SWE-Smith (synthetic, fine) with SWE-Bench-Train and SWE-reBench (real issue text); rows carry no field separating them |
| SWE-bench/SWE-smith-trajectories | MIT, 128 repos | rejected: every trajectory is generated by claude-3-7-sonnet, whose terms do not permit training a competing model |
| R2E-Gym/R2EGym-SFT-Trajectories, r2e-edits/*, ricdomolm/* | no licence tag | rejected: untagged is not permission |

Consequence: the only admissible third-party trajectories come from a five-repository
pool, 72 % of it pandas. Taken whole it would teach a pandas agent, so it is capped
at 1 M tokens per repository (5.66 M -> 3.34 M, 183 -> 68 trajectories) and the
diversity of the training mix comes from our own episodes over the 144 admitted
repository clones.

CORRECTION (same day, after a second audit): an earlier version of this section
said the SWE-smith TASK sets were usable as prompts for our own rollouts. That was
wrong on two counts.

 1. Their problem statements are not mechanically synthesised -- they are written
    by `claude-3-7-sonnet` from the bug diff and failing test (swesmith.com
    /guides/issue_gen). Same generator taint as the trajectories we rejected.
 2. Their repository pool is filtered only for "a license that allows
    non-proprietary use" and explicitly INCLUDES GPL projects, so it is not a
    permissive-only pool.

The SWE-smith PIPELINE is MIT and could be re-run over our own permissive repos
with an open model, but that is building new data, not consuming theirs.

Still open, and it matters because we already accept 3.34 M tokens built on it:
R2E-Gym's SWE-GEN statements are synthesised from commits (not scraped issues),
which is why Nemotron-SWE-v1 was admitted, but WHICH MODEL writes them is not
stated in the README or the env-generation docs. Their published trajectories use
claude-3-5-sonnet; the statement generator is unconfirmed. Flagged as needs-check
rather than assumed clean either way.

Survey conclusion (23 sources): nothing ready-made clears all constraints.
SWE-Gym, Multi-SWE-bench, SWE-PolyBench and SWE-rebench all use scraped GitHub
issue prose. Nemotron-Cascade-SFT-SWE uses an allowed generator (DeepSeek-R1) but
mixed prompt provenance. Our own pipeline is the answer, not a shortcut.

## 11. Provenance chains, audited link by link (2026-09-23)

A permissive tag on the final artifact says nothing about what is inside it. Two
datasets had already taught us this -- SWE-smith is MIT with statements written
by claude-3-7-sonnet, R2E-Gym is Apache-2.0 with an unnamed statement
generator -- so every candidate for the training mix was traced: original source,
intermediate datasets (recursively), EVERY model anywhere in the pipeline
including filters and scorers, whether the publisher held the rights it granted,
and whether human platform content appears anywhere in the chain.

Nearly everything failed. Two patterns account for it.

### 11.1 The Llama Community Licence forbids our exact use

"You will not use the Llama Materials or any output ... to improve any other
large language model (excluding Meta Llama or derivative works)." That is this
project, stated precisely. Contaminated:

  * LLM360/MegaMath -- ~65B of its ~80B synthetic tokens from Llama-3.3-70B,
    disclosed only in arXiv 2504.02807, not on the ODC-By card
  * nvidia/OpenMathInstruct-2 -- Llama-3.1-405B, and NVIDIA's own card states
    the Llama 3.1 licence travels with the data
  * HuggingFaceTB/smoltalk -- Llama-3.1-405B generates the whole Magpie core

**Llama is therefore REMOVED from the approved-generator list.** It had been on
it, which was the same error as assuming "synthetic" meant "not model-written":
an open-weights model is not the same as a permissive output licence.

Note on who is bound: the clause binds whoever accepted Llama's licence. Where a
publisher does not pass it through (MegaMath's ODC-By is silent) a downstream
user arguably never agreed -- the same argument as for Claude-generated
SWE-smith. Where it IS passed through (OpenMathInstruct-2) accepting the data
accepts the clause.

### 11.2 Human platform content re-tagged as permissive

  * nvidia/OpenMathReasoning -- the ENTIRE problem corpus is AoPS forum posts,
    under CC-BY-4.0, with no documented grant from AoPS, whose ToS bars
    commercial exploitation and leaves copyright with the posters
  * open-web-math -- deliberately retains Math Stack Exchange (CC-BY-SA),
    redistributed as ODC-By; inherits into HuggingFaceTB/finemath
  * open-thoughts/OpenThoughts-114k -- FOUR independent broken links under one
    Apache-2.0 tag: AoPS via NuminaMath (rewritten by GPT-4), camel-ai science
    generated entirely by GPT-4, riddle_sense whose own terms forbid
    redistribution, and Codeforces-family problems re-tagged CC-BY-4.0 without a
    visible grant from the problem authors

### 11.3 What survives

  * **HuggingFaceFW/fineweb-2** (ODC-By) -- no LLM anywhere in the pipeline;
    GlotLID and rule-based filters only. Tag matches what is distributed.
  * **HPLT v2.0** -- CC0 scoped explicitly and honestly to the CURATION layer
    (metadata, language labels, dedup decisions), not to the text. Their own
    docs say so. Anyone reading "CC0" as public-domain text is wrong.
  * our own repository clones and episodes.

bigcode/the-stack-v2 has two disclosed structural gaps: licence detection is
`go-license-detector`, a heuristic that fails to detect a licence for ~81% of
repositories and is acknowledged to misclassify some of the ~19% retained; and
opt-outs do NOT propagate backwards into already-published snapshots.

**Consequence: there is no clean third-party mathematics or reasoning corpus.**
For STEM we either generate it ourselves with an Apache-2.0 model, or we do
without.

### 11.4 Our own stack, verified

The Qwen3.5 family -- 4B student, 27B teacher, 35B-A3B and 122B-A10B -- is plain
Apache-2.0, LICENSE text checked directly, with no "improve any other model"
clause. Our generated data is ours.

## 12. Two boundary rulings (2026-09-27)

### 12.1 The FineWeb-Edu classifier's Llama-derived labels: PASSES, but prefer no-model corpora

FineWeb-Edu's quality filter is a BERT regression head trained on 500k
educational-value scores GENERATED BY Llama3-70B-Instruct, then applied to all 15T
FineWeb tokens (threshold >=3 keeps 8% => the 1.3T-token fineweb-edu). The corpus
TEXT is unmodified CommonCrawl; only the keep/discard LABEL derives from Llama.

RULING: passes. The generator rule targets model-WRITTEN text redistributed inside a
corpus. No Llama output is present in the data -- it only informed a selection
decision. This is the same distinction S11.1 drew in REJECTING MegaMath, where ~65B
of 80B tokens ARE Llama-generated prose. Our existing data/fineweb_edu_long.txt
therefore remains valid.

AND, as a stronger standard than strictly required: prefer a corpus with NO model
anywhere in its pipeline for the bulk component.
    tiiuae/falcon-refinedweb   ODC-By 1.0   rule-based "MacroData Refinement", no ML
    HuggingFaceFW/fineweb-2   ODC-By       heuristic + MinHash + langID only
Both are explicitly model-free end to end. fineweb-edu-dedup stays admissible and is
still worth using for its RETAINED METADATA (score, int_score, dump, url), which our
plain-text corpus threw away.

### 12.2 nvidia/OpenMathInstruct-1: ACCEPT the GSM8K-derived rows, and self-generate as well

Licence is a CUSTOM "NVIDIA License", not an SPDX tag. The actual grant text was
read: perpetual, worldwide, royalty-free, right to use/reproduce/prepare
derivatives/sublicense/distribute, NO non-commercial clause, and S3.2 permits
derivatives under different terms -- i.e. permissive and NON-VIRAL.

ACCEPTED, restricted to the GSM8K-derived rows with is_correct == True:
  * solutions generated by Mixtral-8x7B (Apache-2.0) -- an allowed generator, and
    note this PREDATES NVIDIA's switch to Llama-3.1-405B for OpenMathInstruct-2,
    which S11.1 rejected
  * problems are the GSM8K train split: HUMAN-authored under paid contract, MIT
REJECTED: the MATH-derived rows. hendrycks/competition_math is currently
ACCESS-DISABLED on HuggingFace under a takedown, and the aggregated competition
problems carry no documented rights grant -- the same defect as the AoPS case.

ALSO ADOPTED, in parallel: self-generation, because contamination in this domain
lives at the PROBLEM layer, not the solution layer. An LLM asked to invent a maths
problem reproduces memorised ones (this is how TinyGSM fails -- GPT-3.5 wrote its
PROBLEMS); a procedural generator has no original to reproduce and yields exact
answer verification for free. So:
    problems  <- procedural templates, NO model  (google-deepmind/mathematics_dataset,
                 Apache-2.0, fully procedural, is the reusable engine)
    solutions <- the Apache-2.0 35B-A3B Qwen teacher we already serve on llama.cpp,
                 conditioned on (problem, computed answer) -- phrasing and CoT only,
                 never inventing the numbers
Everything generated this way carries a per-row `generator` field, extending the
convention data/episodes/nemotron_swe.jsonl already uses.

### 12.3 Rejected math corpora, for the record

    open-thoughts/OpenThoughts3-1.2M   StackExchange Physics AND Code Golf; math split
                                       likely traces to Llama-3.1-405B
    nvidia/Nemotron-CC-Math            Phi-4 only REFORMATS CommonCrawl maths rather
                                       than regenerating it, so the prose stays the
                                       original authors'. Structurally the
                                       open-web-math failure (S11.2). Distinct from
                                       the ACCEPTED Nemotron-SWE-v1, whose task
                                       statements are synthesised from commit diffs.
    open-r1/OpenR1-Math-220k           aops_forum subset; only synthetic_amc is clean
    nvidia/AceMath-Instruct            CC-BY-NC-4.0
    TinyGSM/TinyGSM                    GPT-3.5 generated the PROBLEMS
    EleutherAI/proof-pile-2            all three subsets fail independently
    HuggingFaceTB/cosmopedia (v1)      wikihow seeds CC-BY-NC-SA; khanacademy seeds
                                       fully copyrighted. v2 dropped both -- use v2.

### 12.4 cosmopedia-v2 needs TWO filters, not one

ODC-By, released text 100% Mixtral-8x7B-Instruct-v0.1 (Apache-2.0); the Llama3-70B
and Qwen1.5-72B generators were ABLATION-ONLY and are not in the release. 28B tokens,
39.1M rows. Schema: prompt, text, token_length, audience, format, seed_data.
  DROP format == "story": v2's 30% "other" bucket borrows v1's stories slice, which
    was seeded from UltraChat and OpenHermes-2.5 -- both GPT-generated, both rejected.
  DROP AutoMathText-seeded rows: AutoMathText is CC-BY-SA-4.0 and built from
    OpenWebMath + RedPajama, re-importing the Math-StackExchange chain S11.2 rejects.
HuggingFace itself flags Mixtral as hallucination-prone on exactly the AutoMathText
maths slice, so cosmopedia is not a maths source regardless.
