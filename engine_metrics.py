from __future__ import annotations

from typing import Dict
import math
import numpy as np

from engine_backtest import BacktestResult
from utils_core import safe_float


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def _norm_cdf(z: float) -> float:
    # Standard normal CDF via erf, no scipy dependency.
    if not np.isfinite(z):
        return float("nan")
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _standardized_moments(x: np.ndarray) -> tuple[float, float]:
    """
    Return (skewness, kurtosis) of x using standardized central moments:
      skew = E[z^3], kurt = E[z^4] where z=(x-mean)/std
    kurtosis is NOT "excess kurtosis" (i.e., normal -> ~3).
    """
    n = int(x.size)
    if n < 4:
        return float("nan"), float("nan")

    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan"), float("nan")

    z = (x - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    return skew, kurt


def probabilistic_sharpe_ratio(
    *,
    sharpe_raw: float,
    n_obs: int,
    skew: float,
    kurt: float,
    sharpe_benchmark_raw: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR), cf. Bailey & López de Prado.
    Computes P(true Sharpe > Sharpe*) under an approximation that accounts for
    non-normality via skewness/kurtosis.

    IMPORTANT:
      - sharpe_raw is NON-annualized: mean(r)/std(r)
      - sharpe_benchmark_raw must use the same scaling (raw, non-annualized)
      - n_obs is number of return observations used for sharpe_raw
    """
    n = int(n_obs)
    sr = float(sharpe_raw)
    sr_star = float(sharpe_benchmark_raw)

    if n < 3 or (not np.isfinite(sr)) or (not np.isfinite(sr_star)):
        return float("nan")

    # If we don't have moments, fall back to normal assumption:
    # denominator = 1 for skew=0, kurt=3
    sk = float(skew) if np.isfinite(skew) else 0.0
    ku = float(kurt) if np.isfinite(kurt) else 3.0

    denom_sq = 1.0 - sk * sr + ((ku - 1.0) / 4.0) * (sr**2)
    if not np.isfinite(denom_sq) or denom_sq <= 0.0:
        return float("nan")

    z = (sr - sr_star) * math.sqrt(max(1.0, float(n - 1))) / math.sqrt(denom_sq)
    return float(_norm_cdf(z))


def compute_metrics(bt: BacktestResult) -> Dict[str, float]:
    """
    Compute a compact, research-focused metric set.

    v2 additions:
      - Sharpe raw (non-annualized) + gross Sharpe
      - Skew/Kurtosis of net returns
      - PSR (probabilistic Sharpe ratio) for benchmark Sharpe*=0 (raw scale)
    """
    r = bt.net_returns.to_numpy(dtype=float)   # net period returns
    g = bt.gross_returns.to_numpy(dtype=float) # gross period returns
    eq = bt.equity_curve.to_numpy(dtype=float)
    n = int(len(r))

    ppy = float(bt.periods_per_year) if np.isfinite(bt.periods_per_year) else 252.0

    mean_r = float(np.mean(r)) if n else float("nan")
    std_r = float(np.std(r, ddof=1)) if n > 1 else float("nan")

    mean_g = float(np.mean(g)) if n else float("nan")
    std_g = float(np.std(g, ddof=1)) if n > 1 else float("nan")

    # Sharpe: keep existing annualized sharpe_net, but also store non-annualized sharpe_raw_*
    sharpe_raw_net = float("nan")
    sharpe_net = float("nan")
    if n > 1 and np.isfinite(std_r) and std_r > 0:
        sharpe_raw_net = mean_r / std_r
        sharpe_net = sharpe_raw_net * math.sqrt(ppy)

    sharpe_raw_gross = float("nan")
    sharpe_gross = float("nan")
    if n > 1 and np.isfinite(std_g) and std_g > 0:
        sharpe_raw_gross = mean_g / std_g
        sharpe_gross = sharpe_raw_gross * math.sqrt(ppy)

    total_return = float(eq[-1] - 1.0) if n else float("nan")
    cagr = float("nan")
    if n > 0 and np.isfinite(eq[-1]) and eq[-1] > 0:
        cagr = float(eq[-1] ** (ppy / n) - 1.0)

    mdd = _max_drawdown(eq)

    turnover = bt.turnover.to_numpy(dtype=float)
    avg_turnover = float(np.mean(turnover)) if n else float("nan")

    # Cost diagnostics
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

    # Higher moments (net)
    skew_net, kurt_net = _standardized_moments(r)

    # PSR for Sharpe*>0 (raw, non-annualized scale)
    psr_net = probabilistic_sharpe_ratio(
        sharpe_raw=sharpe_raw_net,
        n_obs=n,
        skew=skew_net,
        kurt=kurt_net,
        sharpe_benchmark_raw=0.0,
    )

    return {
        # existing keys (keep for backward compatibility)
        "n_obs": float(n),
        "periods_per_year": safe_float(ppy),
        "total_return": safe_float(total_return),
        "cagr": safe_float(cagr),
        "sharpe_net": safe_float(sharpe_net),
        "max_drawdown": safe_float(mdd),
        "avg_turnover": safe_float(avg_turnover),
        "gross_sum": safe_float(gross_total),
        "net_sum": safe_float(net_total),
        "cost_sum": safe_float(cost_total),
        "exposure_abs_mean": safe_float(exposure_abs_mean),
        "exposure_mean": safe_float(exposure_mean),
        "hit_rate": safe_float(hit_rate),

        # new keys
        "sharpe_raw_net": safe_float(sharpe_raw_net),
        "sharpe_gross": safe_float(sharpe_gross),
        "sharpe_raw_gross": safe_float(sharpe_raw_gross),
        "skew_net": safe_float(skew_net),
        "kurt_net": safe_float(kurt_net),
        "psr_net": safe_float(psr_net),
    }
