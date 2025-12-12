from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
import math

import numpy as np
import pandas as pd

from engine_config import BacktestConfig


logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """
    Container for backtest outputs.
    """
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    metrics: Dict[str, float]


def infer_bars_per_year(index: pd.Index) -> float:
    """
    Infer bars_per_year from a DatetimeIndex by using the median step.

    This is best-effort. For custom calendars you should set BacktestConfig.bars_per_year explicitly.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252.0

    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return 252.0

    med = diffs.median()
    seconds = med.total_seconds()
    if seconds <= 0:
        return 252.0

    year_seconds = 365.25 * 24 * 3600
    return float(year_seconds / seconds)


def max_drawdown(equity: pd.Series) -> float:
    """
    Max drawdown as a positive fraction (e.g. 0.25 = -25% drawdown).
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    mdd = float(dd.min())
    return abs(mdd)


def _safe_float(x: float) -> float:
    if x is None:
        return float("nan")
    try:
        x = float(x)
    except Exception:
        return float("nan")
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return x


class Backtester:
    """
    Vectorized single-asset backtester.

    Inputs:
      - prices: price series (close)
      - signals: desired position signals in [-1, 0, 1]
    """
    def run(self, prices: pd.Series, signals: pd.Series, cfg: BacktestConfig) -> BacktestResult:
        if prices is None or signals is None:
            raise ValueError("prices and signals must be provided")

        # Align indices
        idx = prices.index.intersection(signals.index)
        p = prices.loc[idx].astype(float)
        s = signals.loc[idx].astype(float)

        # Compute asset returns
        asset_ret = p.pct_change().fillna(0.0)

        # Apply execution lag to avoid lookahead
        pos = s.shift(int(cfg.execution_lag)).fillna(0.0)

        if not cfg.allow_short:
            pos = pos.clip(lower=0.0, upper=1.0)
        else:
            pos = pos.clip(lower=-1.0, upper=1.0)

        # Transaction costs on position changes
        cost_rate = (float(cfg.fee_bps) + float(cfg.slippage_bps)) / 10_000.0
        delta = pos.diff().fillna(pos)
        costs = delta.abs() * cost_rate

        strat_ret = (pos * asset_ret) - costs
        strat_ret = strat_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        equity = (1.0 + strat_ret).cumprod()
        equity.name = "equity"

        metrics = self._compute_metrics(
            returns=strat_ret,
            equity=equity,
            position=pos,
            bars_per_year=cfg.bars_per_year,
        )

        return BacktestResult(equity=equity, returns=strat_ret, position=pos, metrics=metrics)

    def _compute_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        position: pd.Series,
        bars_per_year: Optional[float],
    ) -> Dict[str, float]:
        if returns.empty:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "vol": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "num_trades": 0.0,
                "win_rate": float("nan"),
                "avg_return": 0.0,
                "exposure": 0.0,
                "turnover": 0.0,
            }

        bpy = float(bars_per_year) if bars_per_year is not None else infer_bars_per_year(returns.index)

        total_return = float(equity.iloc[-1] - 1.0)

        # Annualization
        mean_r = float(returns.mean())
        vol_r = float(returns.std(ddof=0))
        ann_return = (1.0 + mean_r) ** bpy - 1.0 if bpy > 0 else float("nan")
        ann_vol = vol_r * math.sqrt(bpy) if bpy > 0 else float("nan")
        sharpe = (mean_r / vol_r) * math.sqrt(bpy) if vol_r > 0 else 0.0

        # CAGR: use timestamps if possible
        cagr = ann_return
        if isinstance(returns.index, pd.DatetimeIndex) and len(returns.index) >= 2:
            dt = (returns.index[-1] - returns.index[0]).total_seconds()
            years = dt / (365.25 * 24 * 3600) if dt > 0 else None
            if years and years > 0:
                cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)

        mdd = max_drawdown(equity)

        # Trades: count when position changes non-trivially
        delta = position.diff().fillna(position)
        trade_events = (delta.abs() > 1e-12).astype(int)
        num_trades = int(trade_events.sum())

        # Win-rate: fraction of positive returns on bars with exposure
        exposed = position.abs() > 1e-12
        win_rate = float((returns[exposed] > 0).mean()) if exposed.any() else float("nan")

        exposure = float(exposed.mean())
        turnover = float(delta.abs().sum()) / float(len(position)) if len(position) > 0 else 0.0

        return {
            "total_return": _safe_float(total_return),
            "cagr": _safe_float(cagr),
            "ann_return": _safe_float(ann_return),
            "vol": _safe_float(ann_vol),
            "sharpe": _safe_float(sharpe),
            "max_drawdown": _safe_float(mdd),
            "num_trades": float(num_trades),
            "win_rate": _safe_float(win_rate),
            "avg_return": _safe_float(mean_r),
            "exposure": _safe_float(exposure),
            "turnover": _safe_float(turnover),
        }
