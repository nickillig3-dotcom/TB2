"""
tb_paths.py

Central place for all path handling in the TB project.

Constraints (per project spec):
- Repo root is fixed at C:\\TB
- Data / runs are stored on D:\\TB_DATA\\runs\\<run_id>
- No subfolders inside the repo itself for data or runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# --- Root paths (fixed by project specification) ---

# Repository root: all code lives here (flat layout, no subfolders).
REPO_ROOT: Path = Path(r"C:\TB").resolve()

# Data root on D: for market data, runs, logs, models, etc.
DATA_ROOT: Path = Path(r"D:\TB_DATA").resolve()

# All runs will be stored here as D:\TB_DATA\runs\<run_id>\
RUNS_ROOT: Path = DATA_ROOT / "runs"


def get_new_run_id() -> str:
    """
    Generate a new run_id using local time with millisecond resolution.

    Example: '20251209_143015_123'
    """
    now = datetime.now()
    # %f = microseconds; we cut to milliseconds (first 3 digits)
    return now.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def get_run_dir(run_id: str) -> Path:
    """
    Return the directory path for a given run_id (no side effects).
    """
    return RUNS_ROOT / run_id


def ensure_base_dirs() -> None:
    """
    Create base data / runs directories if they do not exist.

    This is idempotent and safe to call multiple times.
    """
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def create_run_dir(run_id: str) -> Path:
    """
    Create and return the directory for a specific run_id.

    Side effects:
        - Ensures D:\\TB_DATA\\runs exists.
        - Creates D:\\TB_DATA\\runs\\<run_id>

    Raises:
        RuntimeError: if the run directory already exists.
    """
    ensure_base_dirs()
    run_dir = get_run_dir(run_id)

    try:
        # parents=False: RUNS_ROOT must already exist (created by ensure_base_dirs)
        # exist_ok=False: fail if the directory already exists (we want unique run_ids)
        run_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Run directory already exists: {run_dir}") from exc

    return run_dir
