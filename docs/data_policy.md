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
