from __future__ import annotations

from typing import Dict

import numpy as np

from engine_backtest import BacktestResult
from utils_core import safe_float


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def compute_metrics(bt: BacktestResult) -> Dict[str, float]:
    """Compute a compact, research-focused metric set.

    Metrics are intentionally minimal in v1 and will expand later (tail risk, stability, regime stats, etc.).
    """
    r = bt.net_returns.to_numpy(dtype=float)
    g = bt.gross_returns.to_numpy(dtype=float)
    eq = bt.equity_curve.to_numpy(dtype=float)

    n = int(len(r))
    ppy = float(bt.periods_per_year) if np.isfinite(bt.periods_per_year) else 252.0

    mean_r = float(np.mean(r)) if n else float("nan")
    std_r = float(np.std(r, ddof=1)) if n > 1 else float("nan")

    sharpe = float("nan")
    if n > 1 and std_r > 0 and np.isfinite(std_r):
        sharpe = (mean_r / std_r) * float(np.sqrt(ppy))

    total_return = float(eq[-1] - 1.0) if n else float("nan")

    cagr = float("nan")
    if n > 0 and eq[-1] > 0:
        cagr = float(eq[-1] ** (ppy / n) - 1.0)

    mdd = _max_drawdown(eq)

    turnover = bt.turnover.to_numpy(dtype=float)
    avg_turnover = float(np.mean(turnover)) if n else float("nan")

    # Approximate cost diagnostics
    tc = bt.cost_components["tc"].to_numpy(dtype=float)
    sl = bt.cost_components["slippage"].to_numpy(dtype=float)
    br = bt.cost_components["borrow"].to_numpy(dtype=float)
    cost_total = float(np.sum(tc + sl + br)) if n else float("nan")

    gross_total = float(np.sum(g)) if n else float("nan")
    net_total = float(np.sum(r)) if n else float("nan")

    # Exposure diagnostics
    pos = bt.positions.to_numpy(dtype=float)
    exposure_abs_mean = float(np.mean(np.abs(pos))) if n else float("nan")
    exposure_mean = float(np.mean(pos)) if n else float("nan")

    # Hit-rate on periods where we have exposure
    active = np.abs(pos) > 1e-12
    hit_rate = float(np.mean(r[active] > 0.0)) if np.any(active) else float("nan")

    return {
        "n_obs": float(n),
        "periods_per_year": safe_float(ppy),
        "total_return": safe_float(total_return),
        "cagr": safe_float(cagr),
        "sharpe_net": safe_float(sharpe),
        "max_drawdown": safe_float(mdd),
        "avg_turnover": safe_float(avg_turnover),
        "gross_sum": safe_float(gross_total),
        "net_sum": safe_float(net_total),
        "cost_sum": safe_float(cost_total),
        "exposure_abs_mean": safe_float(exposure_abs_mean),
        "exposure_mean": safe_float(exposure_mean),
        "hit_rate": safe_float(hit_rate),
    }
