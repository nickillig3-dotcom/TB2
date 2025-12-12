from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Literal
import json
import logging


Mode = Literal["light", "full"]


def _coerce_int(x: Optional[int]) -> Optional[int]:
    if x is None:
        return None
    return int(x)


def _json_default(obj: Any) -> Any:
    # For safe JSON serialization of configs and numpy types.
    try:
        import numpy as np  # local import to keep module lightweight

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


@dataclass
class EngineConfig:
    """
    Global engine/runtime configuration.

    - mode: "light" for weak machines / quick tests, "full" for large runs.
    - random_seed: global seed used across data generation and strategy generation.
    - log_level: standard python logging level name.
    - n_jobs: intended parallelism level (kept =1 by default; the rest of the engine is designed
      so parallel evaluation can be added without redesign).
    """
    mode: Mode = "light"
    random_seed: int = 42
    log_level: str = "INFO"
    n_jobs: Optional[int] = None
    enable_parallel: bool = False

    def apply_mode_defaults(self) -> None:
        if self.n_jobs is None:
            if self.mode == "light":
                self.n_jobs = 1
            else:
                # conservative default: use (cpu_count - 1) but at least 1
                cpu = os.cpu_count() or 1
                self.n_jobs = max(1, cpu - 1)

    def validate(self) -> None:
        if self.mode not in ("light", "full"):
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0")
        # log_level validation is best-effort; python logging will raise on invalid usage later.
        self.apply_mode_defaults()


@dataclass
class DataConfig:
    """
    Data loading configuration.

    price_source:
        - "synthetic": generate a small random-walk price series (default; runs out-of-the-box).
        - "csv": load price from a CSV file.

    max_rows:
        If set, limit rows loaded. StrategyMinerConfig.apply_mode_defaults() will set
        a small default for light mode.
    """
    price_source: str = "synthetic"
    price_csv_path: Optional[str] = None

    # CSV parsing
    timestamp_col: str = "timestamp"
    close_col: str = "close"
    volume_col: Optional[str] = None
    tz: Optional[str] = None

    # Synthetic generation
    synthetic_start: str = "2020-01-01"
    synthetic_freq: str = "D"

    # Resource caps
    max_rows: Optional[int] = None
    tail: bool = True  # if max_rows is set: take tail (recent history) or head

    # Merging & cleaning
    ffill: bool = True

    def apply_mode_defaults(self, mode: Mode) -> None:
        if self.max_rows is None and mode == "light":
            self.max_rows = 3000  # small, but enough for rolling features

    def validate(self) -> None:
        if self.price_source not in ("synthetic", "csv"):
            raise ValueError(f"Unknown price_source: {self.price_source}")
        if self.price_source == "csv" and not self.price_csv_path:
            raise ValueError("price_csv_path must be set when price_source='csv'")
        self.max_rows = _coerce_int(self.max_rows)


@dataclass
class FeatureConfig:
    """
    Feature pipeline defaults (kept deliberately simple).

    The engine itself is generic: features are added by FeatureStep classes.
    This config is only used to build a default pipeline in examples / main.py.
    """
    price_col: str = "price_close"

    return_lags: tuple[int, ...] = (1,)
    rolling_windows: tuple[int, ...] = (5, 20)
    rsi_windows: tuple[int, ...] = (14,)

    dropna: bool = True
    feature_prefix: str = "feat_"

    def validate(self) -> None:
        if not self.price_col:
            raise ValueError("price_col must be non-empty")
        if any(w <= 1 for w in self.rolling_windows):
            raise ValueError("rolling_windows must be > 1")


@dataclass
class BacktestConfig:
    """
    Backtest assumptions.

    fee_bps + slippage_bps are applied on each position change:
        cost = abs(delta_position) * (fee_bps + slippage_bps) / 10_000

    execution_lag:
        Signals at time t are executed at t+execution_lag (default 1) to avoid lookahead.
    """
    price_col: str = "price_close"
    fee_bps: float = 1.0
    slippage_bps: float = 0.5
    execution_lag: int = 1
    bars_per_year: Optional[float] = None
    allow_short: bool = True

    def validate(self) -> None:
        if self.execution_lag < 0:
            raise ValueError("execution_lag must be >= 0")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps/slippage_bps must be >= 0")


@dataclass
class SearchConfig:
    """
    Search space & mining settings.

    max_candidates:
        Total number of strategies to evaluate in a run.

    max_candidates_per_batch:
        Generated per generator call. Useful later when you have many generators.

    feature_prefix:
        Candidate generator will typically search within these columns, e.g. "feat_".
    """
    train_frac: float = 0.7

    max_candidates: Optional[int] = None
    max_candidates_per_batch: Optional[int] = None
    top_k: Optional[int] = None

    # Candidate generation knobs (used by the default RandomThresholdStrategyGenerator)
    feature_prefix: str = "feat_"
    threshold_quantiles: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90)
    direction_choices: tuple[str, ...] = ("trend", "mean_reversion")

    # Simple pruning (speed + reduces junk candidates)
    min_train_trades: int = 5
    min_train_sharpe: float = 0.0
    min_test_sharpe: float = -0.2

    def apply_mode_defaults(self, mode: Mode) -> None:
        if mode == "light":
            if self.max_candidates is None:
                self.max_candidates = 60
            if self.max_candidates_per_batch is None:
                self.max_candidates_per_batch = 30
            if self.top_k is None:
                self.top_k = 10
        else:
            if self.max_candidates is None:
                self.max_candidates = 5000
            if self.max_candidates_per_batch is None:
                self.max_candidates_per_batch = 500
            if self.top_k is None:
                self.top_k = 50

    def validate(self) -> None:
        if not (0.5 <= self.train_frac <= 0.95):
            raise ValueError("train_frac should be in [0.5, 0.95]")
        self.max_candidates = _coerce_int(self.max_candidates)
        self.max_candidates_per_batch = _coerce_int(self.max_candidates_per_batch)
        self.top_k = _coerce_int(self.top_k)


@dataclass
class RobustnessConfig:
    """
    Controls the robustness score composition.

    Keep it intentionally simple; this module is designed to accept more checks later
    (regime splits, bootstraps, Monte-Carlo randomization, etc.).
    """
    w_sharpe: float = 0.5
    w_return: float = 0.3
    w_drawdown: float = 0.2

    # Penalties / soft constraints
    min_test_sharpe_for_full_score: float = 1.0
    max_overfit_sharpe_gap: float = 2.0  # if train_sharpe - test_sharpe > gap, penalize

    def validate(self) -> None:
        s = self.w_sharpe + self.w_return + self.w_drawdown
        if abs(s - 1.0) > 1e-6:
            raise ValueError("Robustness weights must sum to 1.0")


@dataclass
class ResultsConfig:
    """
    Persistent storage settings.

    sqlite_path:
        Lightweight + scalable: good default for a single file in the same folder.

    csv_path:
        Optional export path for human-readable results.
    """
    sqlite_path: str = "strategy_miner_results.sqlite"
    csv_path: Optional[str] = None

    def validate(self) -> None:
        if not self.sqlite_path:
            raise ValueError("sqlite_path must be set")


@dataclass
class StrategyMinerConfig:
    """
    Top-level configuration object for the whole engine.

    Key method: apply_mode_defaults()
        Sets resource caps for light/full mode if they are not explicitly provided.
    """
    engine: EngineConfig = field(default_factory=EngineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    results: ResultsConfig = field(default_factory=ResultsConfig)

    def apply_mode_defaults(self) -> None:
        self.engine.apply_mode_defaults()
        self.data.apply_mode_defaults(self.engine.mode)
        self.search.apply_mode_defaults(self.engine.mode)

    def validate(self) -> None:
        self.apply_mode_defaults()
        self.engine.validate()
        self.data.validate()
        self.features.validate()
        self.backtest.validate()
        self.search.validate()
        self.robustness.validate()
        self.results.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StrategyMinerConfig":
        # Minimal "manual" parsing so we keep explicit dataclasses.
        cfg = StrategyMinerConfig(
            engine=EngineConfig(**d.get("engine", {})),
            data=DataConfig(**d.get("data", {})),
            features=FeatureConfig(**d.get("features", {})),
            backtest=BacktestConfig(**d.get("backtest", {})),
            search=SearchConfig(**d.get("search", {})),
            robustness=RobustnessConfig(**d.get("robustness", {})),
            results=ResultsConfig(**d.get("results", {})),
        )
        cfg.validate()
        return cfg

    @staticmethod
    def from_json(s: str) -> "StrategyMinerConfig":
        return StrategyMinerConfig.from_dict(json.loads(s))


def configure_logging(engine_cfg: EngineConfig) -> None:
    """
    Small helper to configure root logging in a consistent way.
    """
    level = getattr(logging, engine_cfg.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
