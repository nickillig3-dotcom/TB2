from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from engine_config import CvConfig, SplitConfig
from engine_data import DataSet, split_dataset


@dataclass(frozen=True)
class FoldData:
    fold: int
    train: DataSet
    valid: DataSet


@dataclass(frozen=True)
class CvPlan:
    folds: List[FoldData]
    holdout_train: DataSet
    holdout_test: Optional[DataSet]
    meta: dict


def _concat_datasets(a: DataSet, b: DataSet) -> DataSet:
    df = pd.concat([a.frame, b.frame], axis=0)
    return DataSet(df)


def build_cv_plan(ds: DataSet, split_cfg: SplitConfig, cv_cfg: CvConfig) -> CvPlan:
    """
    Unifies:
      - single split -> 1 fold + holdout test
      - walk-forward -> multiple folds + final holdout test

    Purged mechanism:
      - purge_rows removes the last N rows from training right before each validation window.
        This reduces boundary leakage risk (overlapping labels/features windows).
    """
    df = ds.frame
    n = len(df)

    if cv_cfg.method == "single":
        parts = split_dataset(ds, split_cfg)
        folds = [FoldData(fold=0, train=parts["train"], valid=parts["valid"])]
        holdout_train = _concat_datasets(parts["train"], parts["valid"])
        holdout_test = parts["test"]
        meta = {
            "method": "single",
            "n_rows": n,
            "n_folds": 1,
            "fold_ranges": [
                {
                    "fold": 0,
                    "train_t0": str(parts["train"].frame.index[0]),
                    "train_t1": str(parts["train"].frame.index[-1]),
                    "valid_t0": str(parts["valid"].frame.index[0]),
                    "valid_t1": str(parts["valid"].frame.index[-1]),
                }
            ],
            "holdout_t0": str(holdout_test.frame.index[0]),
            "holdout_t1": str(holdout_test.frame.index[-1]),
        }
        return CvPlan(folds=folds, holdout_train=holdout_train, holdout_test=holdout_test, meta=meta)

    # walk-forward
    purge = max(0, int(cv_cfg.purge_rows))

    holdout_test_rows = max(0, int(cv_cfg.holdout_test_rows))
    if holdout_test_rows > 0:
        holdout_start = n - holdout_test_rows
        if holdout_start <= 0:
            raise ValueError("cv.holdout_test_rows too large for dataset length")
        holdout_test = DataSet(df.iloc[holdout_start:n].copy())
    else:
        holdout_start = n
        holdout_test = None

    # keep a gap (purge) before holdout to avoid boundary leakage
    holdout_train_end = max(0, holdout_start - purge)
    holdout_train_start = 0
    if cv_cfg.train_window_rows and cv_cfg.train_window_rows > 0:
        holdout_train_start = max(0, holdout_train_end - int(cv_cfg.train_window_rows))
    holdout_train = DataSet(df.iloc[holdout_train_start:holdout_train_end].copy())

    last_valid_end_limit = holdout_train_end  # folds must end before this

    first_valid_start = int(cv_cfg.min_train_rows) + purge
    if first_valid_start >= last_valid_end_limit:
        raise ValueError(
            "cv config infeasible: first validation start is after (or at) last allowed valid end. "
            "Increase data.n_rows or reduce cv.min_train_rows/holdout_test_rows/purge_rows."
        )

    valid_w = int(cv_cfg.valid_window_rows)
    step = int(cv_cfg.step_rows)
    if valid_w <= 0 or step <= 0:
        raise ValueError("cv.valid_window_rows and cv.step_rows must be > 0")

    # Compute max feasible folds
    max_folds = 0
    cursor = first_valid_start
    while True:
        v0 = cursor
        v1 = v0 + valid_w
        if v1 > last_valid_end_limit:
            break
        max_folds += 1
        cursor += step

    if cv_cfg.n_folds > max_folds:
        raise ValueError(
            f"cv.n_folds={cv_cfg.n_folds} not feasible; max_folds={max_folds}. "
            "Increase data.n_rows or adjust cv.valid_window_rows/step_rows/holdout_test_rows/min_train_rows."
        )

    folds: List[FoldData] = []
    fold_ranges: List[dict] = []

    for i in range(int(cv_cfg.n_folds)):
        v0 = first_valid_start + i * step
        v1 = v0 + valid_w
        if v1 > last_valid_end_limit:
            raise RuntimeError("Internal error: fold exceeds valid end limit")

        train_end = max(0, v0 - purge)
        train_start = 0
        if cv_cfg.train_window_rows and cv_cfg.train_window_rows > 0:
            train_start = max(0, train_end - int(cv_cfg.train_window_rows))

        train_df = df.iloc[train_start:train_end].copy()
        valid_df = df.iloc[v0:v1].copy()

        if len(train_df) < int(cv_cfg.min_train_rows):
            raise ValueError(f"Fold {i}: train too small ({len(train_df)} rows). Increase cv.min_train_rows or n_rows.")
        if len(valid_df) < int(cv_cfg.min_valid_rows):
            raise ValueError(f"Fold {i}: valid too small ({len(valid_df)} rows). Increase cv.min_valid_rows or valid_window_rows.")

        train_ds = DataSet(train_df)
        valid_ds = DataSet(valid_df)

        folds.append(FoldData(fold=i, train=train_ds, valid=valid_ds))
        fold_ranges.append(
            {
                "fold": i,
                "train_idx": [int(train_start), int(train_end)],
                "valid_idx": [int(v0), int(v1)],
                "train_t0": str(train_ds.frame.index[0]),
                "train_t1": str(train_ds.frame.index[-1]),
                "valid_t0": str(valid_ds.frame.index[0]),
                "valid_t1": str(valid_ds.frame.index[-1]),
            }
        )

    meta = {
        "method": "walk_forward",
        "n_rows": n,
        "n_folds": len(folds),
        "purge_rows": purge,
        "holdout_test_rows": holdout_test_rows,
        "holdout_start": holdout_start,
        "holdout_train_end": holdout_train_end,
        "fold_ranges": fold_ranges,
    }
    if holdout_test is not None:
        meta["holdout_t0"] = str(holdout_test.frame.index[0])
        meta["holdout_t1"] = str(holdout_test.frame.index[-1])

    return CvPlan(folds=folds, holdout_train=holdout_train, holdout_test=holdout_test, meta=meta)
