from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Tuple

import numpy as np

from engine_backtest import BacktestResult
from utils_core import safe_float

_NORM = NormalDist()


def _normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    try:
        return float(_NORM.cdf(float(x)))
    except Exception:
        return float("nan")


def _normal_ppf(p: float) -> float:
    """Standard normal inverse CDF (quantile)."""
    try:
        pp = float(p)
    except Exception:
        return float("nan")
    if not np.isfinite(pp):
        return float("nan")
    eps = 1e-12
    pp = min(max(pp, eps), 1.0 - eps)
    return float(_NORM.inv_cdf(pp))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def _moments_z(x: np.ndarray) -> Tuple[float, float]:
    """
    Return (skew, kurtosis) based on standardized z-scores.
    kurtosis is NON-excess (Normal => ~3).
    """
    n = int(x.size)
    if n < 2:
        return float("nan"), float("nan")
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if (not np.isfinite(sigma)) or sigma <= 0.0:
        return float("nan"), float("nan")

    z = (x - mu) / sigma
    skew = float(np.mean(z ** 3)) if n >= 3 else float("nan")
    kurt = float(np.mean(z ** 4)) if n >= 4 else float("nan")
    return skew, kurt


def sharpe_se(sr_per: float, n_obs: int, skew: float, kurt: float) -> float:
    """
    Approx standard error of Sharpe estimator (per-period SR), adjusted for skew/kurt.

    Formula (Bailey & López de Prado style):
      sigma_SR = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1))

    If skew/kurt are missing, falls back to Normal assumption (skew=0, kurt=3).
    """
    n = int(n_obs)
    sr = float(sr_per)
    if n <= 1 or (not np.isfinite(sr)):
        return float("nan")

    sk = float(skew) if np.isfinite(skew) else 0.0
    kt = float(kurt) if np.isfinite(kurt) else 3.0

    denom = float(n - 1)
    term = 1.0 - sk * sr + ((kt - 1.0) / 4.0) * (sr ** 2)
    if (not np.isfinite(term)) or term <= 0.0:
        return float("nan")
    return float(np.sqrt(term / denom))


def probabilistic_sharpe_ratio(sr_per: float, n_obs: int, skew: float, kurt: float, sr0_per: float = 0.0) -> float:
    """
    PSR = P(SR_true > SR0) under approximate normality of SR estimator.
    """
    se = sharpe_se(sr_per, n_obs, skew, kurt)
    if (not np.isfinite(se)) or se <= 0.0:
        return float("nan")
    z = (float(sr_per) - float(sr0_per)) / se
    return _normal_cdf(z)


def deflated_sharpe_ratio(
    sr_per: float,
    n_obs: int,
    skew: float,
    kurt: float,
    trials: int,
    sr0_per: float = 0.0,
) -> float:
    """
    DSR = PSR, but with SR0 replaced by a deflated threshold SR* that accounts for multiple trials.

    We approximate expected max via Blom's approximation:
        p = (M - 0.375) / (M + 0.25)
        z_M = Phi^{-1}(p)

    Then:
        SR* = SR0 + sigma_SR * z_M
        DSR = Phi((SR_hat - SR*) / sigma_SR)
    """
    m = int(trials)
    if m < 1:
        m = 1

    se = sharpe_se(sr_per, n_obs, skew, kurt)
    if (not np.isfinite(se)) or se <= 0.0:
        return float("nan")

    if m == 1:
        sr_star = float(sr0_per)
    else:
        p = (float(m) - 0.375) / (float(m) + 0.25)
        z_m = _normal_ppf(p)
        sr_star = float(sr0_per) + se * z_m

    z = (float(sr_per) - sr_star) / se
    return _normal_cdf(z)


def compute_metrics(bt: BacktestResult) -> Dict[str, float]:
    """
    Compute a compact, research-focused metric set.

    Notes:
    - sharpe_net is ANNUALIZED (using periods_per_year).
    - sharpe_net_per is per-period Sharpe (mean/std on per-period returns).
    - psr_net is PSR for sr0=0 (per-period basis, cache-safe).
    - We do NOT compute dsr_net here, because DSR depends on M=number of trials in the search,
      which is experiment-specific and would poison cached fold metrics.
    """
    r = bt.net_returns.to_numpy(dtype=float)
    g = bt.gross_returns.to_numpy(dtype=float)
    eq = bt.equity_curve.to_numpy(dtype=float)
    n = int(len(r))

    ppy = float(bt.periods_per_year) if np.isfinite(bt.periods_per_year) else 252.0

    mean_r = float(np.mean(r)) if n else float("nan")
    std_r = float(np.std(r, ddof=1)) if n > 1 else float("nan")

    sr_per = float("nan")
    sharpe_ann = float("nan")
    if n > 1 and np.isfinite(std_r) and std_r > 0.0:
        sr_per = float(mean_r / std_r)
        sharpe_ann = float(sr_per * np.sqrt(ppy))

    # Higher moments for significance diagnostics
    skew, kurt = _moments_z(r) if n > 1 else (float("nan"), float("nan"))

    # PSR against 0 Sharpe (cache-safe, does not depend on number of trials)
    psr0 = probabilistic_sharpe_ratio(sr_per, n, skew, kurt, sr0_per=0.0)

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
        "sharpe_net": safe_float(sharpe_ann),
        "sharpe_net_per": safe_float(sr_per),
        "skew_net": safe_float(skew),
        "kurt_net": safe_float(kurt),
        "psr_net": safe_float(psr0),
        "max_drawdown": safe_float(mdd),
        "avg_turnover": safe_float(avg_turnover),
        "gross_sum": safe_float(gross_total),
        "net_sum": safe_float(net_total),
        "cost_sum": safe_float(cost_total),
        "exposure_abs_mean": safe_float(exposure_abs_mean),
        "exposure_mean": safe_float(exposure_mean),
        "hit_rate": safe_float(hit_rate),
    }
