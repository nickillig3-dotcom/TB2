from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from utils_core import stable_hash


@dataclass(frozen=True)
class DataConfig:
    type: str = "synthetic"
    n_rows: int = 400
    n_features: int = 6
    freq: str = "D"
    start: str = "2018-01-01"


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.6
    valid_frac: float = 0.2
    embargo_rows: int = 0
    min_rows: int = 50


@dataclass(frozen=True)
class CvConfig:
    """Cross-validation / robustness split config.

    method:
      - "single": use SplitConfig's single train/valid/test split
      - "walk_forward": multiple chronological folds + final holdout test

    purge_rows:
      - removes last N rows from training before each validation window
        (purged boundary to reduce leakage risk).

    holdout_test_rows:
      - reserves final untouched OOS test slice.
    """
    method: str = "single"  # single|walk_forward
    n_folds: int = 3

    valid_window_rows: int = 50
    step_rows: int = 50

    min_train_rows: int = 120
    train_window_rows: int = 0  # 0 => expanding window, else fixed rolling window
    purge_rows: int = 2

    holdout_test_rows: int = 80
    min_valid_rows: int = 30


@dataclass(frozen=True)
class ResearchConfig:
    n_candidates: int = 20
    generator: str = "linear_random_v1"
    top_k: int = 5
    selection_split: str = "valid"
    selection_metric: str = "sharpe_net"

    # CV scoring: cv_score = agg(valid_metric) - penalty_std * std(valid_metric)
    cv_agg: str = "median"  # median|mean|min
    cv_penalty_std: float = 0.5


@dataclass(frozen=True)
class BacktestConfig:
    execution_lag: int = 1
    leverage: float = 1.0


@dataclass(frozen=True)
class CostConfig:
    tc_bps: float = 2.0
    slippage_bps: float = 1.0
    borrow_bps_annual: float = 0.0


@dataclass(frozen=True)
class PersistenceConfig:
    db_path: str = "strategy_miner.sqlite"


@dataclass(frozen=True)
class ExecutionConfig:
    n_jobs: int = 0  # 0/1 => single-process; >1 => try parallel (optional)
    prefer_processes: bool = True


@dataclass(frozen=True)
class EngineConfig:
    run_name: str = "strategy_miner"
    mode: str = "light"
    seed: int = 42

    data: DataConfig = DataConfig()
    splits: SplitConfig = SplitConfig()
    cv: CvConfig = CvConfig()
    research: ResearchConfig = ResearchConfig()
    backtest: BacktestConfig = BacktestConfig()
    costs: CostConfig = CostConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    execution: ExecutionConfig = ExecutionConfig()

    def config_hash(self) -> str:
        return stable_hash(self)


def _coerce_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _coerce_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def load_config(path: str) -> EngineConfig:
    """Load a JSON config file into dataclasses (minimal validation, future-proof)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data_raw = dict(_get(raw, "data", {}))
    splits_raw = dict(_get(raw, "splits", {}))
    cv_raw = dict(_get(raw, "cv", {}))
    research_raw = dict(_get(raw, "research", {}))
    backtest_raw = dict(_get(raw, "backtest", {}))
    costs_raw = dict(_get(raw, "costs", {}))
    persistence_raw = dict(_get(raw, "persistence", {}))
    execution_raw = dict(_get(raw, "execution", {}))

    cfg = EngineConfig(
        run_name=str(_get(raw, "run_name", "strategy_miner")),
        mode=str(_get(raw, "mode", "light")),
        seed=_coerce_int(_get(raw, "seed", 42), 42),
        data=DataConfig(
            type=str(_get(data_raw, "type", "synthetic")),
            n_rows=_coerce_int(_get(data_raw, "n_rows", 400), 400),
            n_features=_coerce_int(_get(data_raw, "n_features", 6), 6),
            freq=str(_get(data_raw, "freq", "D")),
            start=str(_get(data_raw, "start", "2018-01-01")),
        ),
        splits=SplitConfig(
            train_frac=_coerce_float(_get(splits_raw, "train_frac", 0.6), 0.6),
            valid_frac=_coerce_float(_get(splits_raw, "valid_frac", 0.2), 0.2),
            embargo_rows=_coerce_int(_get(splits_raw, "embargo_rows", 0), 0),
            min_rows=_coerce_int(_get(splits_raw, "min_rows", 50), 50),
        ),
        cv=CvConfig(
            method=str(_get(cv_raw, "method", "single")),
            n_folds=_coerce_int(_get(cv_raw, "n_folds", 3), 3),
            valid_window_rows=_coerce_int(_get(cv_raw, "valid_window_rows", 50), 50),
            step_rows=_coerce_int(_get(cv_raw, "step_rows", 50), 50),
            min_train_rows=_coerce_int(_get(cv_raw, "min_train_rows", 120), 120),
            train_window_rows=_coerce_int(_get(cv_raw, "train_window_rows", 0), 0),
            purge_rows=_coerce_int(_get(cv_raw, "purge_rows", 2), 2),
            holdout_test_rows=_coerce_int(_get(cv_raw, "holdout_test_rows", 80), 80),
            min_valid_rows=_coerce_int(_get(cv_raw, "min_valid_rows", 30), 30),
        ),
        research=ResearchConfig(
            n_candidates=_coerce_int(_get(research_raw, "n_candidates", 20), 20),
            generator=str(_get(research_raw, "generator", "linear_random_v1")),
            top_k=_coerce_int(_get(research_raw, "top_k", 5), 5),
            selection_split=str(_get(research_raw, "selection_split", "valid")),
            selection_metric=str(_get(research_raw, "selection_metric", "sharpe_net")),
            cv_agg=str(_get(research_raw, "cv_agg", "median")),
            cv_penalty_std=_coerce_float(_get(research_raw, "cv_penalty_std", 0.5), 0.5),
        ),
        backtest=BacktestConfig(
            execution_lag=_coerce_int(_get(backtest_raw, "execution_lag", 1), 1),
            leverage=_coerce_float(_get(backtest_raw, "leverage", 1.0), 1.0),
        ),
        costs=CostConfig(
            tc_bps=_coerce_float(_get(costs_raw, "tc_bps", 2.0), 2.0),
            slippage_bps=_coerce_float(_get(costs_raw, "slippage_bps", 1.0), 1.0),
            borrow_bps_annual=_coerce_float(_get(costs_raw, "borrow_bps_annual", 0.0), 0.0),
        ),
        persistence=PersistenceConfig(
            db_path=str(_get(persistence_raw, "db_path", "strategy_miner.sqlite")),
        ),
        execution=ExecutionConfig(
            n_jobs=_coerce_int(_get(execution_raw, "n_jobs", 0), 0),
            prefer_processes=bool(_get(execution_raw, "prefer_processes", True)),
        ),
    )

    # Minimal sanity checks
    if not (0.0 < cfg.splits.train_frac < 1.0):
        raise ValueError("splits.train_frac must be in (0,1)")
    if not (0.0 <= cfg.splits.valid_frac < 1.0):
        raise ValueError("splits.valid_frac must be in [0,1)")
    if cfg.splits.train_frac + cfg.splits.valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must be < 1.0")
    if cfg.backtest.execution_lag < 0:
        raise ValueError("backtest.execution_lag must be >= 0")
    if cfg.research.n_candidates <= 0:
        raise ValueError("research.n_candidates must be > 0")
    if cfg.data.n_rows <= 10:
        raise ValueError("data.n_rows must be > 10")
    if cfg.data.n_features <= 0:
        raise ValueError("data.n_features must be > 0")
    if cfg.research.top_k <= 0:
        raise ValueError("research.top_k must be > 0")

    if cfg.cv.method not in {"single", "walk_forward"}:
        raise ValueError("cv.method must be 'single' or 'walk_forward'")
    if cfg.cv.n_folds < 1:
        raise ValueError("cv.n_folds must be >= 1")
    if cfg.cv.valid_window_rows < 1:
        raise ValueError("cv.valid_window_rows must be >= 1")
    if cfg.cv.step_rows < 1:
        raise ValueError("cv.step_rows must be >= 1")
    if cfg.cv.min_train_rows < 10:
        raise ValueError("cv.min_train_rows must be >= 10")
    if cfg.cv.purge_rows < 0:
        raise ValueError("cv.purge_rows must be >= 0")
    if cfg.cv.train_window_rows < 0:
        raise ValueError("cv.train_window_rows must be >= 0")

    if cfg.research.cv_agg not in {"median", "mean", "min"}:
        raise ValueError("research.cv_agg must be one of: median|mean|min")
    if cfg.research.cv_penalty_std < 0:
        raise ValueError("research.cv_penalty_std must be >= 0")

    return cfg
