from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

from utils_core import stable_hash


def _drop_none(obj: Any) -> Any:
    """Recursively drop None values from dict/list structures (keeps empty lists/dicts)."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            vv = _drop_none(v)
            if vv is None:
                continue
            out[str(k)] = vv
        return out
    if isinstance(obj, list):
        out_list: list[Any] = []
        for x in obj:
            xx = _drop_none(x)
            if xx is None:
                continue
            out_list.append(xx)
        return out_list
    return obj


def _coerce_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _coerce_int_opt(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _coerce_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _coerce_str_opt(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _coerce_bool_opt(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _coerce_str_list_opt(x: Any) -> Optional[list[str]]:
    if x is None:
        return None
    if isinstance(x, list):
        out: list[str] = []
        for it in x:
            s = _coerce_str_opt(it)
            if s:
                out.append(s)
        return out or None
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",")]
        parts = [p for p in parts if p]
        return parts or None
    return None


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


@dataclass(frozen=True)
class DataConfig:
    # core selector
    type: str = "synthetic"

    # synthetic generator params
    n_rows: int = 400
    n_features: int = 6
    freq: str = "D"
    start: str = "2018-01-01"

    # identity overrides (optional)
    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None

    # file/csv params (optional)
    path: Optional[str] = None
    timestamp_col: Optional[str] = None
    price_col: Optional[str] = None
    feature_cols: Optional[list[str]] = None
    tz: Optional[str] = None
    row_limit: Optional[int] = None
    dropna: Optional[bool] = None
    dedup: Optional[bool] = None
    sep: Optional[str] = None


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.6
    valid_frac: float = 0.2
    embargo_rows: int = 0
    min_rows: int = 50


@dataclass(frozen=True)
class CvConfig:
    """
    Cross-validation / robustness split config.

    method:
      - "single": use SplitConfig's single train/valid/test split
      - "walk_forward": multiple chronological folds + final holdout test

    purge_rows:
      - removes last N rows from training before each validation window (purged boundary)

    holdout_test_rows:
      - reserves final untouched OOS test slice
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
        # Important: drop None so schema extensions don't invalidate old hashes.
        return stable_hash(_drop_none(asdict(self)))


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
            dataset_id=_coerce_str_opt(_get(data_raw, "dataset_id", None)),
            dataset_version=_coerce_str_opt(_get(data_raw, "dataset_version", None)),
            path=_coerce_str_opt(_get(data_raw, "path", None)),
            timestamp_col=_coerce_str_opt(_get(data_raw, "timestamp_col", None)),
            price_col=_coerce_str_opt(_get(data_raw, "price_col", None)),
            feature_cols=_coerce_str_list_opt(_get(data_raw, "feature_cols", None)),
            tz=_coerce_str_opt(_get(data_raw, "tz", None)),
            row_limit=_coerce_int_opt(_get(data_raw, "row_limit", None)),
            dropna=_coerce_bool_opt(_get(data_raw, "dropna", None)),
            dedup=_coerce_bool_opt(_get(data_raw, "dedup", None)),
            sep=_coerce_str_opt(_get(data_raw, "sep", None)),
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

    # --- Sanity checks ---
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
    if cfg.research.top_k <= 0:
        raise ValueError("research.top_k must be > 0")

    if cfg.data.n_features <= 0:
        raise ValueError("data.n_features must be > 0")

    if cfg.data.type == "synthetic":
        if cfg.data.n_rows <= 10:
            raise ValueError("data.n_rows must be > 10 for synthetic")
    elif cfg.data.type == "csv":
        if not cfg.data.path:
            raise ValueError("data.path is required for data.type='csv'")

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
