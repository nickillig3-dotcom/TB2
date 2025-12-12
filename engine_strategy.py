from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from engine_config import ResearchConfig
from utils_core import stable_hash


@dataclass(frozen=True)
class StrategySpec:
    """Pure-data strategy description (persistable, hashable)."""
    name: str
    params: Dict[str, Any]

    def spec_hash(self) -> str:
        return stable_hash(self)


class Strategy:
    """Executable strategy object.

    Contract:
      - generate_positions(df) returns a pandas Series aligned with df.index.
      - positions are *target exposures* (e.g., -1..+1) before leverage.
    """
    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class LinearSignalStrategy(Strategy):
    """Minimal, non-retail toy strategy: linear comb of arbitrary features.

    This is intentionally simple to validate the engine pipeline; it is NOT an end-goal strategy.
    """
    def __init__(self, weights: np.ndarray, threshold: float = 0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.threshold = float(threshold)

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        feat_cols = [c for c in df.columns if c.startswith("f")]
        if len(feat_cols) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} feature columns, got {len(feat_cols)} "
                f"(features: {feat_cols})"
            )
        x = df[feat_cols].to_numpy(dtype=float, copy=False)
        score = x.dot(self.weights)
        # target position: {-1,0,+1} based on threshold band
        pos = np.zeros(len(df), dtype=float)
        pos[score > self.threshold] = 1.0
        pos[score < -self.threshold] = -1.0
        return pd.Series(pos, index=df.index, name="pos")


def build_strategy(spec: StrategySpec) -> Strategy:
    if spec.name == "linear_signal_v1":
        w = np.asarray(spec.params["weights"], dtype=float)
        thr = float(spec.params.get("threshold", 0.0))
        return LinearSignalStrategy(weights=w, threshold=thr)
    raise ValueError(f"Unknown strategy name: {spec.name}")


def generate_candidates(
    cfg: ResearchConfig,
    n_features: int,
    rng: np.random.Generator,
) -> Iterable[StrategySpec]:
    """Candidate generator (StrategySpecs) for research loops.

    Today: simple random linear weights for pipeline validation.
    Tomorrow: meta-DSL, grammar search, genetic programming, Bayesian opt, etc.
    """
    if cfg.generator != "linear_random_v1":
        raise ValueError(f"Unknown generator: {cfg.generator}")

    for _ in range(cfg.n_candidates):
        w = rng.normal(size=(n_features,))
        w = (w / (np.linalg.norm(w) + 1e-12)).tolist()
        thr = float(abs(rng.normal(scale=0.25)))  # small dead-zone
        yield StrategySpec(
            name="linear_signal_v1",
            params={"weights": w, "threshold": thr},
        )
