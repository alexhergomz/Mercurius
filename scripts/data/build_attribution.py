"""Attribution manifest and NOTICE for everything the model trains on.

Permissive licences are not "no obligations": MIT, BSD and Apache-2.0 all
require the copyright notice and licence to travel with substantial portions
of the work, and CC-BY requires attribution to the dataset's creators. Our
training sequences embed real repository text, so the obligation is real and
the answer is to record provenance per item and ship a NOTICE.

Reads every episode file (ours and the filtered third-party ones) plus the
clone manifest, and writes:

  data/ATTRIBUTION.md  one row per repository: licence, commit, URL, episodes,
                       and one row per third-party dataset: licence, source,
                       generator, citation
  data/NOTICE          the short form to ship with a model release

    python scripts/build_attribution.py
"""
import json
import os
from collections import defaultdict

from mercurius.paths import DATA_DIR, ROOT

DATASET_INFO = {
    "nvidia/Nemotron-SWE-v1": {
        "license": "CC-BY-4.0 (trajectories); repository content under each repo's own licence",
        "tasks_from": "R2E-Gym/R2E-Gym-Subset (Apache-2.0) -- problem statements "
                      "synthesised from commits by the SWE-GEN pipeline, not copied from issues",
        "generator": "Qwen3-Coder-480B (Apache-2.0)",
    },
}


CORPORA = [
    ("HuggingFaceFW/fineweb-edu", "ODC-BY", "training",
     "general web text; attribution required by ODC-BY"),
    ("data/fineweb_edu_long.txt", "ODC-BY (derived)", "training",
     "long documents filtered from FineWeb-Edu"),
    ("data/synth_recall_4b.txt", "ODC-BY (derived)", "training",
     "synthetic retention documents built from FineWeb-Edu sentences"),
    ("Salesforce/wikitext (WikiText-2)", "CC-BY-SA-3.0 / GFDL", "EVALUATION ONLY",
     "share-alike: never trained on, only perplexity is measured"),
    ("emozilla/pg19-test (PG-19 test books)", "no tag on the mirror; the books "
     "themselves are pre-1919 public domain", "EVALUATION ONLY",
     "length-extrapolation measurements"),
    ("sgoel9/paul_graham_essays (RULER haystack)", "mirror tagged MIT, but the "
     "essays are the author's copyright", "EVALUATION ONLY",
     "needle-in-haystack filler; never trained on. Replace with public-domain "
     "text before distributing any evaluation artifact"),
]

MODELS = [
    ("Qwen/Qwen3.5-4B", "Apache-2.0", "student; this model is a MODIFIED version of it"),
    ("Qwen/Qwen3.5-27B", "Apache-2.0", "distillation teacher; its outputs are used as "
     "training targets, which its licence permits"),
    ("unsloth/Qwen3.5-27B-GGUF", "Apache-2.0", "4-bit teacher used to generate episodes"),
]

MODIFICATIONS = [
    "RMSNorm gains folded into the consuming projections; the folded norms "
    "replaced by ScaleNorm (one scalar each)",
    "GatedDeltaNet layers lifted to Kimi Delta Attention and then to Gated "
    "DeltaNet-2 (channel-wise erase/write gates)",
    "rotary position encoding removed (NoPE)",
    "K/V projections of the full-attention layers replaced by a shared "
    "low-rank latent (MLA), grouped by measured retrieval role",
    "per-head query maps added to the attention layers",
    "weights quantized to NF4; adapters (VeRA) and the new weights kept in "
    "higher precision",
    "recovery by distillation from Qwen3.5-27B",
]


def main():
    ep_dir = DATA_DIR / "episodes"
    repos, datasets, files = defaultdict(lambda: {"episodes": 0, "tokens": 0}), defaultdict(int), {}
    for fn in sorted(os.listdir(ep_dir)):
        if not fn.endswith(".jsonl"):
            continue
        n = 0
        for line in open(ep_dir / fn):
            r = json.loads(line)
            n += 1
            key = r.get("repo") or "(unknown)"
            repos[key]["episodes"] += 1
            repos[key]["tokens"] += r.get("n_tokens", 0)
            repos[key]["license"] = r.get("license")
            repos[key]["commit"] = r.get("commit") or repos[key].get("commit")
            if r.get("dataset_id"):
                datasets[r["dataset_id"]] += 1
                repos[key]["via"] = r["dataset_id"]
        files[fn] = n
    clones = json.load(open(ROOT / "data/repos/clones.json"))
    for name, m in clones.items():
        if name in repos and not repos[name].get("commit"):
            repos[name]["commit"] = m.get("commit")
            repos[name]["license"] = repos[name].get("license") or m.get("license")

    out = [f"# Attribution\n",
           "Everything below is used under a licence permitting commercial use "
           "without share-alike terms (docs/data_policy.md). Repository content "
           "keeps its own licence; the copyright notices inside the files are "
           "left intact in every training sequence.\n",
           "## Episode files\n",
           "| file | episodes |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in files.items()]
    out += ["\n## Third-party trajectory datasets\n"]
    for d, cnt in sorted(datasets.items()):
        info = DATASET_INFO.get(d, {})
        out += [f"### {d} ({cnt} trajectories)", "",
                f"- Licence: {info.get('license', 'see dataset card')}",
                f"- Tasks from: {info.get('tasks_from', 'see dataset card')}",
                f"- Generated by: {info.get('generator', 'see dataset card')}",
                f"- Source: https://huggingface.co/datasets/{d}", ""]
    out += ["\n## Corpora\n", "| source | licence | use | note |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in CORPORA]
    out += ["\n## Models\n", "| model | licence | role |", "|---|---|---|"]
    out += [f"| [{a}](https://huggingface.co/{a}) | {b} | {c} |" for a, b, c in MODELS]
    out += ["\n## Modifications to Qwen3.5-4B (Apache-2.0 section 4(b))\n"]
    out += [f"- {m}" for m in MODIFICATIONS]
    out += [f"\n## Repositories ({len(repos)})\n",
            "| repository | licence | commit | episodes | tokens | via |", "|---|---|---|---|---|---|"]
    for name, m in sorted(repos.items(), key=lambda kv: -kv[1]["tokens"]):
        out.append(f"| [{name}](https://github.com/{name}) | {m.get('license') or '?'} | "
                   f"`{(m.get('commit') or '')[:10]}` | {m['episodes']} | {m['tokens']:,} | "
                   f"{m.get('via', 'built here')} |")
    (DATA_DIR / "ATTRIBUTION.md").write_text("\n".join(out) + "\n")

    notice = ["This model is a modified version of Qwen/Qwen3.5-4B (Apache License 2.0),",
              "distilled from Qwen/Qwen3.5-27B (Apache License 2.0).",
              "", "Modifications made to the original model:"]
    notice += [f"  - {m}" for m in MODIFICATIONS]
    notice += ["",
              "This model was trained in part on content from open-source software",
              "repositories and on agent trajectories derived from them.", "",
              "Repository content is used under each project's own licence (MIT,",
              "Apache-2.0, BSD-2/3-Clause, ISC, Unlicense, CC0 and similar). Copyright",
              "notices within that content are preserved. The full list of repositories,",
              "with licence and commit, is in ATTRIBUTION.md.", "",
              "Third-party trajectory datasets used, with their licences:"]
    for d in sorted(datasets):
        notice.append(f"  - {d}: {DATASET_INFO.get(d, {}).get('license', 'see dataset card')}")
    notice += ["", "Corpora:"]
    notice += [f"  - {a}: {b} ({c})" for a, b, c, _ in CORPORA]
    notice += ["", "No content was used from repositories whose owners opted out of The",
               "Stack, and no GitHub issue or pull-request prose was used: task",
               "statements are either generated from code or synthesised.", ""]
    (DATA_DIR / "NOTICE").write_text("\n".join(notice))
    print(f"{len(repos)} repositories, {len(datasets)} third-party datasets, "
          f"{sum(files.values())} episodes")
    print(f"-> {DATA_DIR / 'ATTRIBUTION.md'}\n-> {DATA_DIR / 'NOTICE'}")


if __name__ == "__main__":
    main()
