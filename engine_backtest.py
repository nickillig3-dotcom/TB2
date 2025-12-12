from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from engine_config import BacktestConfig, CostConfig
from utils_core import ensure_monotonic_datetime_index, infer_periods_per_year


@dataclass(frozen=True)
class BacktestResult:
    net_returns: pd.Series
    gross_returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    cost_components: Dict[str, pd.Series]
    equity_curve: pd.Series
    periods_per_year: float


def run_backtest(
    df: pd.DataFrame,
    raw_positions: pd.Series,
    bt_cfg: BacktestConfig,
    cost_cfg: CostConfig,
) -> BacktestResult:
    """Vectorized single-instrument backtest with basic cost model.

    Assumptions:
      - 'price' column exists
      - raw_positions are based on information available at time t
      - execution_lag shifts positions forward to avoid lookahead
    """
    ensure_monotonic_datetime_index(df)

    if "price" not in df.columns:
        raise ValueError("df must contain a 'price' column")

    price = df["price"].astype(float)
    if price.isna().any():
        raise ValueError("price contains NaN values")

    if not raw_positions.index.equals(df.index):
        raw_positions = raw_positions.reindex(df.index)

    raw_positions = raw_positions.astype(float).fillna(0.0)

    # Shift positions forward for execution (anti-lookahead)
    lag = int(bt_cfg.execution_lag)
    if lag < 0:
        raise ValueError("execution_lag must be >= 0")

    positions = raw_positions.shift(lag).fillna(0.0) * float(bt_cfg.leverage)

    # Returns
    ret = price.pct_change().fillna(0.0)

    gross = positions * ret

    # Turnover = absolute position change (proxy for traded notional)
    turnover = positions.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(positions.iloc[0])  # first trade from flat

    # Costs (bps on turnover)
    tc = turnover * (float(cost_cfg.tc_bps) / 10000.0)
    slip = turnover * (float(cost_cfg.slippage_bps) / 10000.0)

    # Borrow cost per period (bps annualized on abs exposure)
    ppy = infer_periods_per_year(df.index)
    borrow_per_period = (float(cost_cfg.borrow_bps_annual) / 10000.0) / max(1.0, ppy)
    borrow = positions.abs() * borrow_per_period

    net = gross - tc - slip - borrow

    equity = (1.0 + net).cumprod()
    equity.name = "equity"

    return BacktestResult(
        net_returns=net.rename("ret_net"),
        gross_returns=gross.rename("ret_gross"),
        positions=positions.rename("pos_exec"),
        turnover=turnover.rename("turnover"),
        cost_components={
            "tc": tc.rename("cost_tc"),
            "slippage": slip.rename("cost_slippage"),
            "borrow": borrow.rename("cost_borrow"),
        },
        equity_curve=equity,
        periods_per_year=ppy,
    )
