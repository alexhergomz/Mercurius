"""Filesystem layout, in one place and overridable.

Every path the pipeline touches resolves from a single root so the tree can be
relocated or pointed at a scratch disk without editing code. Set MERCURIUS_ROOT
to move everything; set an individual variable to move just that.

The original tree hardcoded ~90 absolute paths across 29 modules, which is why
this exists.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "MERCURIUS_ROOT", Path(__file__).resolve().parents[2])).resolve()

CKPT_DIR = Path(os.environ.get("MERCURIUS_CKPT", ROOT / "ckpt"))
DATA_DIR = Path(os.environ.get("MERCURIUS_DATA", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("MERCURIUS_MODELS", ROOT / "models"))
CACHE_DIR = Path(os.environ.get("MERCURIUS_CACHE", ROOT / "cache"))
LOGS_DIR = Path(os.environ.get("MERCURIUS_LOGS", ROOT / "logs"))

# the unmodified teacher, as downloaded
BASE_MODEL = MODELS_DIR / "qwen3.5-0.8b"
# output of surgery stages A+B: norms fused, GatedDeltaNet lifted to KDA
STAGE_AB = CKPT_DIR / "qwen3.5-0.8b-stageAB"

WIKITEXT = DATA_DIR / "wikitext.txt"       # held-out eval
FINEWEB = DATA_DIR / "fineweb_edu.txt"     # training / calibration corpus

# Writes refuse below this much free space. The Jetson's root filesystem is the
# only filesystem; filling it bricks the board.
MIN_FREE_GB = float(os.environ.get("MERCURIUS_MIN_FREE_GB", "8.0"))


def ensure_dirs():
    for d in (CKPT_DIR, CACHE_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_free_space(need_gb=MIN_FREE_GB):
    """Refuse rather than discover a full disk at write time."""
    import shutil
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < need_gb:
        raise SystemExit(f"refusing to write: {free:.1f} GB free, "
                         f"need {need_gb} GB headroom")
    return free
