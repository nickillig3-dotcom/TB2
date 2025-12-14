from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Tuple

import numpy as np

from engine_backtest import BacktestResult
from utils_core import safe_float

_NORM = NormalDist()


def _normal_cdf(x: float) -> float:
    try:
        return float(_NORM.cdf(float(x)))
    except Exception:
        return float("nan")


def _normal_ppf(p: float) -> float:
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


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """
    Fast rolling std (ddof=0) via cumulative sums.
    Output has NaN for the first window-1 points.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    out = np.full(n, np.nan, dtype=float)
    w = int(window)
    if w <= 1 or n < w:
        return out

    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)

    sum1 = c1[w - 1 :] - np.concatenate(([0.0], c1[: n - w]))
    sum2 = c2[w - 1 :] - np.concatenate(([0.0], c2[: n - w]))

    mean = sum1 / float(w)
    var = (sum2 / float(w)) - mean * mean
    var = np.maximum(var, 0.0)

    out[w - 1 :] = np.sqrt(var)
    return out


def _ann_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=float)
    n = int(r.size)
    if n <= 1:
        return float("nan")
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if (not np.isfinite(sd)) or sd <= 0.0:
        return float("nan")
    return float((mu / sd) * np.sqrt(float(periods_per_year)))


def _infer_reference_returns_from_bt(
    gross_returns: np.ndarray,
    positions: np.ndarray,
    *,
    eps_pos: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Try to infer underlying/reference returns from the backtest itself.

    Assumption (typical): gross_return[t] ~= position[t-1] * underlying_return[t]
    => underlying_return[t] ~= gross_return[t] / position[t-1]

    Returns:
      ref_returns_filled (np.ndarray, length n)
      coverage (fraction of points where inference was possible)
    """
    g = np.asarray(gross_returns, dtype=float)
    p = np.asarray(positions, dtype=float)
    n = int(g.size)
    if n < 3 or p.size != n:
        return np.zeros(n, dtype=float), 0.0

    p_prev = np.roll(p, 1)
    p_prev[0] = np.nan

    good = np.isfinite(g) & np.isfinite(p_prev) & (np.abs(p_prev) > float(eps_pos))
    coverage = float(np.mean(good)) if n else 0.0

    ref = np.full(n, np.nan, dtype=float)
    ref[good] = g[good] / p_prev[good]

    # To keep rolling_std stable we fill non-inferred points with 0.
    # We only USE this ref series if coverage is high enough.
    ref_filled = np.where(np.isfinite(ref), ref, 0.0)
    return ref_filled, coverage


def _regime_metrics(
    strategy_returns: np.ndarray,
    source_returns: np.ndarray,
    periods_per_year: float,
    *,
    window: int = 20,
    q_low: float = 0.33,
    q_high: float = 0.66,
    min_obs_per_regime: int = 30,
) -> Dict[str, float]:
    """
    Regime split based on rolling volatility of SOURCE returns.
    Sharpe is computed on STRATEGY returns inside those regimes.

    NEW: adaptive effective min_obs_per_regime to avoid "always fallback" in short folds:
         - For small samples: allow >=10 obs per regime
         - For large samples: cap at 30 (so Full-Mode remains robust)
    """
    r = np.asarray(strategy_returns, dtype=float)
    src = np.asarray(source_returns, dtype=float)
    n = int(r.size)
    out: Dict[str, float] = {}

    out["regime_window"] = float(window)
    out["regime_q_low"] = float(q_low)
    out["regime_q_high"] = float(q_high)

    # Approx number of valid vol points
    vol_n_est = max(0, n - int(window) + 1)
    cap = int(min_obs_per_regime) if int(min_obs_per_regime) > 0 else 30
    min_obs_eff = int(min(cap, max(10, int(0.2 * float(vol_n_est)))))
    out["regime_min_obs_per_regime"] = float(min_obs_eff)

    if n < max(int(window) + 2, min_obs_eff):
        out["regime_used_fallback"] = 1.0
        out["regime_vol_q33"] = float("nan")
        out["regime_vol_q66"] = float("nan")
        out["regime_n_low"] = 0.0
        out["regime_n_mid"] = 0.0
        out["regime_n_high"] = 0.0
        out["sharpe_regime_low_net"] = float("nan")
        out["sharpe_regime_mid_net"] = float("nan")
        out["sharpe_regime_high_net"] = float("nan")
        out["regime_worst_sharpe_net"] = float("nan")
        out["regime_best_sharpe_net"] = float("nan")
        out["regime_spread_sharpe_net"] = float("nan")
        out["regime_low_frac"] = float("nan")
        out["regime_mid_frac"] = float("nan")
        out["regime_high_frac"] = float("nan")
        return out

    vol = _rolling_std(src, window=window)
    valid = np.isfinite(vol)
    vol_valid = vol[valid]
    if vol_valid.size < min_obs_eff:
        out["regime_used_fallback"] = 1.0
        out["regime_vol_q33"] = float("nan")
        out["regime_vol_q66"] = float("nan")
        out["regime_n_low"] = 0.0
        out["regime_n_mid"] = 0.0
        out["regime_n_high"] = 0.0
        out["sharpe_regime_low_net"] = float("nan")
        out["sharpe_regime_mid_net"] = float("nan")
        out["sharpe_regime_high_net"] = float("nan")
        out["regime_worst_sharpe_net"] = float("nan")
        out["regime_best_sharpe_net"] = float("nan")
        out["regime_spread_sharpe_net"] = float("nan")
        out["regime_low_frac"] = float("nan")
        out["regime_mid_frac"] = float("nan")
        out["regime_high_frac"] = float("nan")
        return out

    q1 = float(np.quantile(vol_valid, q_low))
    q2 = float(np.quantile(vol_valid, q_high))
    out["regime_vol_q33"] = q1
    out["regime_vol_q66"] = q2

    if (not np.isfinite(q1)) or (not np.isfinite(q2)) or (q2 <= q1):
        mask_low = np.zeros(n, dtype=bool)
        mask_high = np.zeros(n, dtype=bool)
        mask_mid = valid.copy()
    else:
        mask_low = valid & (vol <= q1)
        mask_mid = valid & (vol > q1) & (vol <= q2)
        mask_high = valid & (vol > q2)

    def _metric_for_mask(mask: np.ndarray) -> Tuple[float, int]:
        rr = r[mask]
        nn = int(rr.size)
        if nn < min_obs_eff:
            return float("nan"), nn
        return _ann_sharpe(rr, periods_per_year), nn

    s_low, n_low = _metric_for_mask(mask_low)
    s_mid, n_mid = _metric_for_mask(mask_mid)
    s_high, n_high = _metric_for_mask(mask_high)

    out["regime_used_fallback"] = 0.0
    out["regime_n_low"] = float(n_low)
    out["regime_n_mid"] = float(n_mid)
    out["regime_n_high"] = float(n_high)

    out["sharpe_regime_low_net"] = float(s_low)
    out["sharpe_regime_mid_net"] = float(s_mid)
    out["sharpe_regime_high_net"] = float(s_high)

    total_reg = float(n_low + n_mid + n_high)
    if total_reg > 0:
        out["regime_low_frac"] = float(n_low / total_reg)
        out["regime_mid_frac"] = float(n_mid / total_reg)
        out["regime_high_frac"] = float(n_high / total_reg)
    else:
        out["regime_low_frac"] = float("nan")
        out["regime_mid_frac"] = float("nan")
        out["regime_high_frac"] = float("nan")

    sharpe_list = [x for x in [s_low, s_mid, s_high] if np.isfinite(x)]
    if sharpe_list:
        worst = float(min(sharpe_list))
        best = float(max(sharpe_list))
        out["regime_worst_sharpe_net"] = worst
        out["regime_best_sharpe_net"] = best
        out["regime_spread_sharpe_net"] = float(best - worst)
    else:
        out["regime_used_fallback"] = 1.0
        out["regime_worst_sharpe_net"] = float("nan")
        out["regime_best_sharpe_net"] = float("nan")
        out["regime_spread_sharpe_net"] = float("nan")

    return out


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

    skew, kurt = _moments_z(r) if n > 1 else (float("nan"), float("nan"))
    psr0 = probabilistic_sharpe_ratio(sr_per, n, skew, kurt, sr0_per=0.0)

    total_return = float(eq[-1] - 1.0) if n else float("nan")

    cagr = float("nan")
    if n > 0 and eq[-1] > 0:
        cagr = float(eq[-1] ** (ppy / n) - 1.0)

    mdd = _max_drawdown(eq)

    turnover = bt.turnover.to_numpy(dtype=float)
    avg_turnover = float(np.mean(turnover)) if n else float("nan")

    tc = bt.cost_components["tc"].to_numpy(dtype=float)
    sl = bt.cost_components["slippage"].to_numpy(dtype=float)
    br = bt.cost_components["borrow"].to_numpy(dtype=float)
    cost_total = float(np.sum(tc + sl + br)) if n else float("nan")

    gross_total = float(np.sum(g)) if n else float("nan")
    net_total = float(np.sum(r)) if n else float("nan")

    pos = bt.positions.to_numpy(dtype=float)
    exposure_abs_mean = float(np.mean(np.abs(pos))) if n else float("nan")
    exposure_mean = float(np.mean(pos)) if n else float("nan")

    active = np.abs(pos) > 1e-12
    hit_rate = float(np.mean(r[active] > 0.0)) if np.any(active) else float("nan")

    metrics = {
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

    # ---- Regime Robustness (market-driven proxy if possible) ----
    ref_src, coverage = _infer_reference_returns_from_bt(g, pos, eps_pos=1e-6)

    # Use inferred reference only if it covers most points AND has non-trivial variance.
    use_ref = (coverage >= 0.90) and (np.isfinite(ref_src).all()) and (float(np.std(ref_src)) > 1e-12)

    metrics["regime_ref_used"] = safe_float(1.0 if use_ref else 0.0)
    metrics["regime_ref_coverage"] = safe_float(coverage)

    src = ref_src if use_ref else r

    reg = _regime_metrics(
        strategy_returns=r,
        source_returns=src,
        periods_per_year=ppy,
        window=20,
        q_low=0.33,
        q_high=0.66,
        min_obs_per_regime=30,
    )
    for k, v in reg.items():
        metrics[k] = safe_float(v)

    # If worst is NaN, fallback to overall sharpe so selection doesn't break.
    if not np.isfinite(float(metrics.get("regime_worst_sharpe_net", float("nan")))):
        metrics["regime_used_fallback"] = 1.0
        metrics["regime_worst_sharpe_net"] = safe_float(sharpe_ann)
        metrics["regime_best_sharpe_net"] = safe_float(sharpe_ann)
        metrics["regime_spread_sharpe_net"] = safe_float(0.0)

    return metrics
