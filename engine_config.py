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
class ResearchConfig:
    n_candidates: int = 20
    generator: str = "linear_random_v1"
    top_k: int = 5
    selection_split: str = "valid"  # train|valid|test
    selection_metric: str = "sharpe_net"  # metric key computed by engine_metrics


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
    mode: str = "light"  # light|full (only config-level differences)
    seed: int = 42

    data: DataConfig = DataConfig()
    splits: SplitConfig = SplitConfig()
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
        research=ResearchConfig(
            n_candidates=_coerce_int(_get(research_raw, "n_candidates", 20), 20),
            generator=str(_get(research_raw, "generator", "linear_random_v1")),
            top_k=_coerce_int(_get(research_raw, "top_k", 5), 5),
            selection_split=str(_get(research_raw, "selection_split", "valid")),
            selection_metric=str(_get(research_raw, "selection_metric", "sharpe_net")),
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

    # Minimal sanity checks (keep lightweight; move to richer validators later)
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

    return cfg
