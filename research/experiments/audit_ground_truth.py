"""Cross-check the parser's ground truth against an independent grep, per language.

The first pilot lost a whole repository (MAVSDK, 0/3) to a ground-truth bug: the
`impact` answer omitted C/C++ header declarations, so correct answers were
rejected AND -- worse -- an answer could only be accepted by omitting the header
too, which would have trained the student to forget them. A wrong oracle is more
damaging than no oracle, because it is silent.

That bug class is not specific to C. Java interfaces, Go interface method sets,
Rust trait signatures and TypeScript .d.ts files all DECLARE without defining.
This audits every language we parse, by asking a completely different tool the
same question: grep for the bare identifier, and compare file sets.

Disagreement is not automatically a bug -- grep matches comments, strings and
unrelated same-named symbols, so grep is a SUPERSET. What matters is the
direction: files grep finds that the parser missed are candidate bugs, and are
printed for inspection.

    python experiments/audit_ground_truth.py --per-lang 3
"""
import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict

from mercurius.data.code_graph import LANGS, CodeGraph, make_tasks
from mercurius.paths import ROOT

EXT_OF = defaultdict(list)
for ext, lang in LANGS.items():
    EXT_OF[lang].append(ext)

# grep hits that are NOT a use of the symbol
NOISE = re.compile(r"^\s*(//|#|\*|/\*|--)|^\s*$")


def grep_files(root, symbol, exts):
    args = ["grep", "-rIl", "--binary-files=without-match"]
    args += [f"--include=*{e}" for e in exts]
    args += [rf"\b{re.escape(symbol)}\b", root]
    r = subprocess.run(args, capture_output=True, text=True)
    return {os.path.relpath(p, root) for p in r.stdout.split() if p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=3, help="repos per language")
    ap.add_argument("--symbols", type=int, default=6, help="symbols per repo")
    ap.add_argument("--out", default=str(ROOT / "logs/ground_truth_audit.json"))
    a = ap.parse_args()
    clones = {k: v for k, v in json.load(open(ROOT / "data/repos/clones.json")).items()
              if "error" not in v}
    yields = json.load(open(ROOT / "data/repos/graph_yield.json"))
    names = sorted((n for n in clones if yields.get(n)), key=lambda n: -yields[n])

    seen_lang, report = Counter(), []
    for name in names:
        root = os.path.join(ROOT, clones[name]["path"])
        if not os.path.isdir(root):
            continue
        try:
            g = CodeGraph(root)
        except Exception:
            continue
        cands = g.unique_defs()
        if not cands:
            continue
        lang = cands[0][1][3]
        if seen_lang[lang] >= a.per_lang:
            continue
        seen_lang[lang] += 1
        exts = EXT_OF[lang]
        miss_tot = hit_tot = 0
        examples = []
        for sym, (path, line, kind, lg), sites in cands[:a.symbols]:
            ours = {p for p, _ in sites} | {path} | set(g.decls.get(sym, ()))
            theirs = grep_files(root, sym, exts)
            missed = theirs - ours
            hit_tot += len(ours)
            miss_tot += len(missed)
            if missed:
                examples.append({"symbol": sym, "missed": sorted(missed)[:4],
                                 "ours": sorted(ours)[:4]})
        rate = miss_tot / max(hit_tot + miss_tot, 1)
        report.append({"repo": name, "lang": lang, "symbols": min(len(cands), a.symbols),
                       "files_in_answers": hit_tot, "files_grep_found_we_missed": miss_tot,
                       "miss_rate": round(rate, 3), "examples": examples[:3]})
        print(f"{lang:<12} {name:<40} miss {miss_tot:>3}/{hit_tot + miss_tot:<3} "
              f"({rate:.0%})", flush=True)
        for e in examples[:2]:
            print(f"     {e['symbol']}: missed {e['missed']}")
    json.dump(report, open(a.out, "w"), indent=1)
    worst = sorted(report, key=lambda r: -r["miss_rate"])[:5]
    print("\nworst by miss rate:")
    for r in worst:
        print(f"  {r['lang']:<12} {r['repo']:<40} {r['miss_rate']:.0%}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
