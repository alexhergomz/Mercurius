"""Port a module from the original flat research tree into the package.

The research tree at MERCURIUS_ORIGIN is still where experiments run, so it
drifts ahead. This re-applies the same rewrite the one-shot migration did --
flat imports to dotted, absolute paths to mercurius.paths, sys.path bootstrap
removed -- so a ported file lands in the same shape as everything around it.

Deliberately not a git subtree or symlink: the two trees are meant to diverge
(the original keeps its hardcoded paths and its history), and a file should only
move when someone decides it is ready to.
"""
import argparse
import os
import re
import shutil
from pathlib import Path

ORIGIN = Path(os.environ.get("MERCURIUS_ORIGIN", "/home/srdelam/qwen-surgery"))
ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "kda_model": "mercurius.legacy.models.kda",
    "quantize": "mercurius.legacy.models.quantize",
    "stage_a_fuse": "mercurius.legacy.surgery.norm_fusion",
    "stage_b_lift": "mercurius.legacy.surgery.kda_lift",
    "stage_c_dial": "mercurius.legacy.surgery.rope_dial",
    "stage_d_transmla": "mercurius.legacy.surgery.transmla",
    "lora": "mercurius.legacy.adapters.lora",
    "layerscale": "mercurius.legacy.adapters.layerscale",
    "ssmax": "mercurius.legacy.adapters.ssmax",
    "run_care": "mercurius.legacy.calibration.care",
    "attn_covs": "mercurius.legacy.calibration.attention",
    "logit_cache": "mercurius.legacy.recovery.logit_cache",
    "train_recovery": "mercurius.legacy.recovery.train",
    "train_transfer": "mercurius.legacy.recovery.transfer",
    "evalsuite": "mercurius.legacy.eval.suite",
    "characterize": "mercurius.legacy.eval.characterize",
    "eval_longctx": "mercurius.legacy.eval.longctx",
    "eval_compressed_retrieval": "mercurius.legacy.eval.compressed_retrieval",
    "harness": "mercurius.legacy.harness",
    "guard": "mercurius.guard",
    # architecture
    "gdn2_model": "mercurius.legacy.models.gdn2",
    "mtp_conv": "mercurius.legacy.models.mtp_conv",
    "perhead_q": "mercurius.legacy.surgery.perhead_q",
    # data
    "synth_recall": "mercurius.legacy.recovery.synth_recall",
    # evaluation
    "eval_retrieval_ab": "mercurius.legacy.eval.retrieval_ab",
    "ruler_gen": "mercurius.legacy.eval.ruler_gen",
    "ruler_eval": "mercurius.legacy.eval.ruler",
    "nolex_data": "mercurius.legacy.eval.nolex_data",
    "nolex": "mercurius.legacy.eval.nolex",
    "verify_checkpoint": "mercurius.legacy.eval.verify_checkpoint",
    "retrieval_heads": "mercurius.legacy.eval.retrieval_heads",
    "alloc_screen": "mercurius.legacy.eval.alloc_screen",
}

PATHS = [
    (f'"{ORIGIN}/ckpt/qwen3.5-0.8b-stageAB"', "str(STAGE_AB)"),
    (f'"{ORIGIN}/models/qwen3.5-0.8b"', "str(BASE_MODEL)"),
    (f'"{ORIGIN}/data/wikitext.txt"', "str(WIKITEXT)"),
    (f'"{ORIGIN}/data/fineweb_edu.txt"', "str(FINEWEB)"),
    (f'"{ORIGIN}/ckpt/adapters-combined.pt"', "str(CKPT_DIR / 'adapters-combined.pt')"),
    (f'"{ORIGIN}/cache/kv_covs.pt"', "str(CACHE_DIR / 'kv_covs.pt')"),
    (f'"{ORIGIN}/ckpt"', "str(CKPT_DIR)"),
    (f'"{ORIGIN}/cache"', "str(CACHE_DIR)"),
    (f'"{ORIGIN}/logs"', "str(LOGS_DIR)"),
    (f'"{ORIGIN}/data"', "str(DATA_DIR)"),
    (f'"{ORIGIN}/data/fineweb_edu_long.txt"', "str(DATA_DIR / 'fineweb_edu_long.txt')"),
    (f'"{ORIGIN}/data/synth_recall.txt"', "str(DATA_DIR / 'synth_recall.txt')"),
    (f'"{ORIGIN}/data/ruler/PaulGrahamEssays.json"',
     "str(DATA_DIR / 'ruler' / 'PaulGrahamEssays.json')"),
    (f'"{ORIGIN}/data/nolima"', "str(DATA_DIR / 'nolima')"),
    (f"{ORIGIN}/", ""),
]
NAMES = ["STAGE_AB", "BASE_MODEL", "WIKITEXT", "FINEWEB",
         "CKPT_DIR", "CACHE_DIR", "LOGS_DIR", "DATA_DIR"]


def rewrite(text):
    text = re.sub(r'^\s*sys\.path\.insert\([^)]*\)\s*\n', '', text, flags=re.M)
    # longest first: "eval_longctx" must not be clobbered by a match on "eval"
    for old in sorted(MODULES, key=len, reverse=True):
        new = MODULES[old]
        # match anywhere on the line, so in-FUNCTION imports are caught too --
        # anchoring on line start missed 15 of them during the migration, each
        # one a runtime failure that import-checking cannot see
        text = re.sub(rf'(?m)^(\s*)from {re.escape(old)} import ',
                      rf'\1from {new} import ', text)
        text = re.sub(rf'(?m)^import {re.escape(old)}$',
                      f'from {new.rsplit(".", 1)[0]} import '
                      f'{new.rsplit(".", 1)[1]} as {old}', text)

    used = set()
    for lit, repl in PATHS:
        if lit in text:
            text = text.replace(lit, repl)
            used.update(n for n in NAMES if n in repl)
    if used:
        text = _insert_import(
            text, f"from mercurius.paths import {', '.join(sorted(used))}")
    return text


def _insert_import(text, stmt):
    """Place stmt after the leading import block, using the AST for the boundary.

    Line scanning gets this wrong on a multi-line parenthesized import: it sees
    the opening line, inserts after it, and splits the continuation off from its
    own statement -- a SyntaxError. ast.end_lineno knows where a statement
    actually ends. Falls back to line 0 if the file will not parse.
    """
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return stmt + "\n" + text
    end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = max(end, node.end_lineno or node.lineno)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                      # module docstring
        elif end:
            break                         # first real statement after imports
    lines = text.split("\n")
    lines.insert(end, stmt)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+",
                    help="flat module names, e.g. lora train_recovery")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for name in a.names:
        src = ORIGIN / "src" / f"{name}.py"
        if not src.exists():
            print(f"  {name}: not in {src}"); continue
        dotted = MODULES.get(name)
        if dotted:
            dst = ROOT / "src" / Path(dotted.replace(".", "/") + ".py")
        else:
            dst = ROOT / "experiments" / f"{name}.py"
        new = rewrite(src.read_text())
        if a.dry_run:
            old = dst.read_text() if dst.exists() else ""
            import difflib
            d = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                          str(dst), "new", lineterm="", n=0))
            print(f"  {name} -> {dst.relative_to(ROOT)}: {len(d)} diff lines")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(new)
            print(f"  {name} -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
