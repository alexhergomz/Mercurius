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
