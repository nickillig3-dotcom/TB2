from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd

from engine_config import FeatureConfig


logger = logging.getLogger(__name__)


class FeatureStep(ABC):
    """
    A single feature engineering step that adds (or modifies) columns.

    Design goals:
    - Steps can be composed in a FeaturePipeline.
    - Steps may have a lookback (history requirement) for correct test-time computation.
    - Steps may store fitted parameters (e.g., scaling) by implementing fit().
    """
    name: str = "feature_step"
    lookback: int = 0  # required history bars

    def fit(self, df: pd.DataFrame) -> "FeatureStep":
        # Default: stateless.
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class PriceReturnsStep(FeatureStep):
    """
    Adds return and log-return features for a price column.
    """
    price_col: str
    lags: Sequence[int] = (1,)
    out_prefix: str = "feat_"

    name: str = "price_returns"

    @property
    def lookback(self) -> int:
        return int(max(self.lags)) if self.lags else 1

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df.columns:
            raise KeyError(f"Missing price_col '{self.price_col}' for PriceReturnsStep")

        price = df[self.price_col].astype(float)

        for lag in self.lags:
            lag = int(lag)
            df[f"{self.out_prefix}ret_{lag}"] = price.pct_change(lag)
            df[f"{self.out_prefix}logret_{lag}"] = np.log(price).diff(lag)
        return df


@dataclass
class RollingStatsStep(FeatureStep):
    """
    Adds rolling mean/std features for a given base column.
    """
    base_col: str
    windows: Sequence[int] = (5, 20)
    out_prefix: str = "feat_"
    min_periods: Optional[int] = None

    name: str = "rolling_stats"

    @property
    def lookback(self) -> int:
        return int(max(self.windows)) if self.windows else 1

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.base_col not in df.columns:
            raise KeyError(f"Missing base_col '{self.base_col}' for RollingStatsStep")

        s = df[self.base_col].astype(float)
        for w in self.windows:
            w = int(w)
            mp = self.min_periods if self.min_periods is not None else w
            df[f"{self.out_prefix}{self.base_col}_mean_{w}"] = s.rolling(w, min_periods=mp).mean()
            df[f"{self.out_prefix}{self.base_col}_std_{w}"] = s.rolling(w, min_periods=mp).std(ddof=0)
        return df


@dataclass
class RSIStep(FeatureStep):
    """
    Adds RSI (Relative Strength Index) on log returns.

    RSI is a simple momentum/mean-reversion indicator; used here only as a feature example.
    """
    logret_col: str = "feat_logret_1"
    window: int = 14
    out_col: str = "feat_rsi_14"

    name: str = "rsi"

    @property
    def lookback(self) -> int:
        return int(self.window) + 1

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.logret_col not in df.columns:
            raise KeyError(f"Missing logret_col '{self.logret_col}' for RSIStep")

        r = df[self.logret_col].astype(float).fillna(0.0)
        gain = r.clip(lower=0.0)
        loss = (-r).clip(lower=0.0)

        # Wilder's smoothing via EMA with alpha=1/window
        alpha = 1.0 / float(self.window)
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / (avg_loss.replace(0.0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df[self.out_col] = rsi
        return df


@dataclass
class ZScoreStep(FeatureStep):
    """
    Adds a rolling z-score feature for a base column.

    This is stateless and uses only past values, so it is safe in train/test.
    """
    base_col: str
    window: int = 20
    out_col: str = "feat_z_20"
    min_periods: Optional[int] = None

    name: str = "zscore"

    @property
    def lookback(self) -> int:
        return int(self.window)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.base_col not in df.columns:
            raise KeyError(f"Missing base_col '{self.base_col}' for ZScoreStep")

        s = df[self.base_col].astype(float)
        mp = self.min_periods if self.min_periods is not None else self.window
        mean = s.rolling(self.window, min_periods=mp).mean()
        std = s.rolling(self.window, min_periods=mp).std(ddof=0)
        df[self.out_col] = (s - mean) / std.replace(0.0, np.nan)
        return df


@dataclass
class StandardizeStep(FeatureStep):
    """
    Standardizes selected feature columns using train-set mean/std.

    This step demonstrates how to avoid leakage:
      - fit() computes mean/std on TRAIN only
      - transform() applies to any df (train/test)

    NOTE: Standardization is optional and not used by default in light-mode examples,
    because many simple rule strategies don't need it.
    """
    cols: Sequence[str]
    out_prefix: str = "feat_std_"

    name: str = "standardize"
    lookback: int = 0

    means_: Optional[pd.Series] = None
    stds_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "StandardizeStep":
        cols = [c for c in self.cols if c in df.columns]
        if not cols:
            self.means_ = pd.Series(dtype=float)
            self.stds_ = pd.Series(dtype=float)
            return self
        self.means_ = df[cols].mean()
        self.stds_ = df[cols].std(ddof=0).replace(0.0, np.nan)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("StandardizeStep must be fit() before transform().")

        cols = [c for c in self.cols if c in df.columns]
        for c in cols:
            df[f"{self.out_prefix}{c}"] = (df[c] - self.means_.get(c, 0.0)) / self.stds_.get(c, np.nan)
        return df


class FeaturePipeline:
    """
    Applies a list of FeatureStep objects to build a feature-rich DataFrame.

    Design goals:
    - Sequential composition: later steps can depend on features created by earlier steps.
    - No leakage-ready: steps can implement fit() (train-only) and transform() (train/test).
    - Efficient: fit_transform only computes features once on the train set.

    Important properties:
    - .lookback: maximum history bars required to compute features correctly on a test slice.
    """
    def __init__(self, steps: Sequence[FeatureStep], dropna: bool = True):
        self.steps: List[FeatureStep] = list(steps)
        self.dropna = bool(dropna)
        self._is_fit = False

    @property
    def lookback(self) -> int:
        return int(max((getattr(s, "lookback", 0) for s in self.steps), default=0))

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """
        Fit all steps on the training slice only.

        Note: Some steps may need intermediate features from earlier steps to fit correctly.
        Therefore we execute transforms sequentially while fitting, but we do NOT return
        the transformed dataframe here (use fit_transform for that).
        """
        df = train_df.copy()
        for step in self.steps:
            step.fit(df)
            df = step.transform(df)
        self._is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform any dataframe using already-fitted steps.
        """
        if not self._is_fit:
            logger.warning("FeaturePipeline.transform() called before fit(). Assuming stateless steps.")
        out = df.copy()
        for step in self.steps:
            out = step.transform(out)
        if self.dropna:
            out = out.dropna()
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on train_df and return the transformed train features (computed once).
        """
        df = train_df.copy()
        for step in self.steps:
            step.fit(df)
            df = step.transform(df)
        self._is_fit = True
        if self.dropna:
            df = df.dropna()
        return df

    def describe(self) -> List[str]:
        """
        Human-readable pipeline overview.
        """
        return [f"{i:02d} - {s.__class__.__name__} (lookback={getattr(s, 'lookback', 0)})" for i, s in enumerate(self.steps)]


def build_default_pipeline(cfg: FeatureConfig) -> FeaturePipeline:
    """
    Convenience factory for a minimal-but-extensible default pipeline.

    It generates:
      - returns/logreturns
      - rolling stats of logret_1
      - z-score of logret_1
      - RSI on logret_1

    You can replace this factory later with a config-driven registry of feature steps.
    """
    steps: List[FeatureStep] = []
    steps.append(PriceReturnsStep(price_col=cfg.price_col, lags=cfg.return_lags, out_prefix=cfg.feature_prefix))

    # Use logret_1 as base for rolling features if present
    base_logret = f"{cfg.feature_prefix}logret_1"
    steps.append(RollingStatsStep(base_col=base_logret, windows=cfg.rolling_windows, out_prefix=""))

    # Z-score as a common "normalized momentum" feature
    z_out = f"{cfg.feature_prefix}z_logret1_{max(cfg.rolling_windows)}"
    steps.append(ZScoreStep(base_col=base_logret, window=max(cfg.rolling_windows), out_col=z_out))

    # RSI
    rsi_out = f"{cfg.feature_prefix}rsi_{cfg.rsi_windows[0]}"
    steps.append(RSIStep(logret_col=base_logret, window=cfg.rsi_windows[0], out_col=rsi_out))

    return FeaturePipeline(steps=steps, dropna=cfg.dropna)
