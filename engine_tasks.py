from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from engine_backtest import run_backtest
from engine_config import BacktestConfig, CostConfig
from engine_metrics import compute_metrics
from engine_strategy import StrategySpec, build_strategy


# ---- Global worker context (set once per experiment) -------------------------

_EVAL_CONTEXT: Dict[str, Any] = {}


def init_eval_context(context: Dict[str, Any]) -> None:
    """
    Initializer for worker processes (and also called in main process for threads/single).

    Expected keys:
      - "folds": list[tuple[pd.DataFrame, pd.DataFrame]]  # (train_df, valid_df)
      - "holdout_train": pd.DataFrame
      - "holdout_test": Optional[pd.DataFrame]
    """
    global _EVAL_CONTEXT
    _EVAL_CONTEXT = context


def _get_fold_data(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds = _EVAL_CONTEXT.get("folds")
    if folds is None:
        raise RuntimeError("Eval context not initialized (missing 'folds').")
    return folds[int(fold)]


def _get_holdout() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    ht = _EVAL_CONTEXT.get("holdout_train")
    if ht is None:
        raise RuntimeError("Eval context not initialized (missing 'holdout_train').")
    return ht, _EVAL_CONTEXT.get("holdout_test")


# ---- Tasks ------------------------------------------------------------------

@dataclass(frozen=True)
class FoldEvalTask:
    task_id: str
    fold: int
    strategy_spec: StrategySpec
    strategy_hash: str
    strategy_json: str
    backtest_cfg: BacktestConfig
    cost_cfg: CostConfig


@dataclass(frozen=True)
class HoldoutEvalTask:
    task_id: str
    strategy_spec: StrategySpec
    strategy_hash: str
    strategy_json: str
    backtest_cfg: BacktestConfig
    cost_cfg: CostConfig


def eval_fold_task(task: FoldEvalTask) -> Dict[str, Any]:
    """Evaluate one strategy on one fold: train + valid."""
    train_df, valid_df = _get_fold_data(task.fold)

    out: Dict[str, Any] = {
        "task_id": task.task_id,
        "fold": int(task.fold),
        "strategy_name": task.strategy_spec.name,
        "strategy_hash": task.strategy_hash,
        "strategy_json": task.strategy_json,
        "train_n": int(len(train_df)),
        "valid_n": int(len(valid_df)),
    }

    # Build/fit once
    try:
        strat = build_strategy(task.strategy_spec).fit(train_df)
    except Exception as e:
        err = repr(e)
        out["train"] = {"status": "error", "error": err, "metrics": {}, "elapsed_ms": 0}
        out["valid"] = {"status": "error", "error": err, "metrics": {}, "elapsed_ms": 0, "artifact": None}
        return out

    # Train
    t0 = time.perf_counter()
    try:
        raw_pos_train = strat.generate_positions(train_df)
        bt_train = run_backtest(train_df, raw_pos_train, task.backtest_cfg, task.cost_cfg)
        m_train = compute_metrics(bt_train)
        train_res = {"status": "ok", "error": None, "metrics": m_train}
    except Exception as e:
        train_res = {"status": "error", "error": repr(e), "metrics": {}}
    train_res["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    out["train"] = train_res

    # Valid
    t1 = time.perf_counter()
    try:
        raw_pos_valid = strat.generate_positions(valid_df)
        bt_valid = run_backtest(valid_df, raw_pos_valid, task.backtest_cfg, task.cost_cfg)
        m_valid = compute_metrics(bt_valid)

        artifact = {
            "t0": str(valid_df.index[0]),
            "t1": str(valid_df.index[-1]),
            "equity_end": float(bt_valid.equity_curve.iloc[-1]),
            "equity_head": bt_valid.equity_curve.iloc[: min(50, len(bt_valid.equity_curve))].to_list(),
        }
        valid_res = {"status": "ok", "error": None, "metrics": m_valid, "artifact": artifact}
    except Exception as e:
        valid_res = {"status": "error", "error": repr(e), "metrics": {}, "artifact": None}
    valid_res["elapsed_ms"] = int((time.perf_counter() - t1) * 1000)
    out["valid"] = valid_res

    return out


def eval_holdout_task(task: HoldoutEvalTask) -> Dict[str, Any]:
    """Evaluate a strategy on final holdout test after fitting on holdout_train."""
    holdout_train, holdout_test = _get_holdout()
    if holdout_test is None:
        return {
            "task_id": task.task_id,
            "strategy_name": task.strategy_spec.name,
            "strategy_hash": task.strategy_hash,
            "strategy_json": task.strategy_json,
            "test": {"status": "error", "error": "holdout_test is None", "metrics": {}, "elapsed_ms": 0, "artifact": None},
        }

    out: Dict[str, Any] = {
        "task_id": task.task_id,
        "strategy_name": task.strategy_spec.name,
        "strategy_hash": task.strategy_hash,
        "strategy_json": task.strategy_json,
        "test_n": int(len(holdout_test)),
    }

    try:
        strat = build_strategy(task.strategy_spec).fit(holdout_train)
    except Exception as e:
        out["test"] = {"status": "error", "error": repr(e), "metrics": {}, "elapsed_ms": 0, "artifact": None}
        return out

    t0 = time.perf_counter()
    try:
        raw_pos_test = strat.generate_positions(holdout_test)
        bt_test = run_backtest(holdout_test, raw_pos_test, task.backtest_cfg, task.cost_cfg)
        m_test = compute_metrics(bt_test)
        artifact = {
            "t0": str(holdout_test.index[0]),
            "t1": str(holdout_test.index[-1]),
            "equity_end": float(bt_test.equity_curve.iloc[-1]),
            "equity_head": bt_test.equity_curve.iloc[: min(50, len(bt_test.equity_curve))].to_list(),
        }
        test_res = {"status": "ok", "error": None, "metrics": m_test, "artifact": artifact}
    except Exception as e:
        test_res = {"status": "error", "error": repr(e), "metrics": {}, "artifact": None}
    test_res["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    out["test"] = test_res
    return out
