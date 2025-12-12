from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Abstract strategy interface.

    The engine treats strategies as "signal generators":
      generate_signals(features) -> pd.Series in [-1, 0, +1] (or continuous later)

    Signals are later turned into positions by the backtest module (with execution lag etc.).
    """
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def to_config(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class FeatureThresholdStrategy(Strategy):
    """
    Minimal example strategy (used only to test the engine end-to-end):

    - Select one feature column.
    - If feature > threshold -> long
    - If feature < -threshold -> short (if allow_short)
    - Else -> flat

    direction:
      - "trend": follow the sign of the feature (momentum)
      - "mean_reversion": invert the sign of the feature

    NOTE:
      This is intentionally simple. The focus is on the research infrastructure.
      Real StrategyGenerators can later produce trees, programs, ML models, ensembles, etc.
    """
    feature_col: str
    threshold: float
    direction: str = "trend"
    allow_short: bool = True
    min_hold_bars: int = 1
    neutral_zone: float = 0.0  # extra dead-zone around 0.0 to reduce churn

    @property
    def name(self) -> str:
        return f"FeatureThreshold({self.direction},{self.feature_col},thr={self.threshold:.4g})"

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        if self.feature_col not in features.columns:
            raise KeyError(f"Strategy requires missing feature_col '{self.feature_col}'")

        x = features[self.feature_col].astype(float)
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        thr = float(self.threshold)
        nz = float(self.neutral_zone)

        long_mask = x > (thr + nz)
        short_mask = x < -(thr + nz)

        raw = np.zeros(len(x), dtype=float)
        raw[long_mask.values] = 1.0
        if self.allow_short:
            raw[short_mask.values] = -1.0

        if self.direction == "mean_reversion":
            raw = -raw
        elif self.direction != "trend":
            raise ValueError(f"Unknown direction: {self.direction}")

        sig = pd.Series(raw, index=features.index, name="signal")

        # Optional holding constraint: keep previous non-zero signal for N bars
        # (kept simple; default min_hold_bars=1 means no extra work)
        hold = int(self.min_hold_bars)
        if hold > 1:
            sig = _apply_min_hold(sig, hold)

        return sig

    def to_config(self) -> Dict[str, Any]:
        return {
            "type": "FeatureThresholdStrategy",
            "feature_col": self.feature_col,
            "threshold": float(self.threshold),
            "direction": self.direction,
            "allow_short": bool(self.allow_short),
            "min_hold_bars": int(self.min_hold_bars),
            "neutral_zone": float(self.neutral_zone),
        }


def _apply_min_hold(sig: pd.Series, hold_bars: int) -> pd.Series:
    """
    Enforces a minimum hold time on a discrete signal series.

    This is a small helper, implemented in a straightforward way.
    In full-scale mode you would likely use a more optimized approach.
    """
    hold_bars = int(max(1, hold_bars))
    if hold_bars <= 1:
        return sig

    arr = sig.to_numpy(copy=True)
    out = arr.copy()

    last = 0.0
    remaining = 0
    for i in range(len(arr)):
        if remaining > 0:
            # Keep last position, ignore new signals until hold time passes
            out[i] = last
            remaining -= 1
            continue

        if arr[i] != 0.0:
            last = arr[i]
            out[i] = last
            remaining = hold_bars - 1
        else:
            out[i] = 0.0
            last = 0.0
            remaining = 0

    return pd.Series(out, index=sig.index, name=sig.name)
