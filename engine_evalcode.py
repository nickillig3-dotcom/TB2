from __future__ import annotations

from utils_core import compute_files_hash

# Evaluation-semantics files only (cache key).
# Excludes: research_run.py, engine_persistence.py, engine_executor.py, engine_research.py
EVAL_CODE_FILES: list[str] = [
    "engine_backtest.py",
    "engine_metrics.py",
    "engine_cv.py",
    "engine_data.py",
    "engine_strategy.py",
    "engine_tasks.py",
    "utils_core.py",
]


def compute_eval_code_hash() -> str:
    """
    Hash of evaluation-critical code only. Used for cache stability when
    non-eval code changes (CLI, persistence, executor, etc.).
    """
    return compute_files_hash(EVAL_CODE_FILES)
