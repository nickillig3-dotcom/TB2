#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm_backtest.py — Strategy Miner Backtest Core (Perpetual Futures)

What this file provides:
- Loads OHLCV from the Parquet dataset created by sm_data.py
- Aligns multi-market data on the intersection of timestamps
- Vectorized portfolio backtest with:
  - next-open execution model (open-to-open, last bar open-to-close)
  - fees + slippage via turnover (|Δposition|)
  - optional funding (if stored in data/parquet/{exchange}/funding/{symbol}/...)
  - leverage constraints (gross + per-asset caps)
- Robust performance metrics + monthly return stats

This is research tooling. No profitability guarantees.

Requires:
  pip install -U numpy polars pyarrow
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("Missing dependency: numpy. Install via: pip install -U numpy") from e

try:
    import polars as pl
except Exception as e:
    raise RuntimeError("Missing dependency: polars. Install via: pip install -U polars") from e


# -----------------------------
# Paths + Logging
# -----------------------------
@dataclass(frozen=True)
class Paths:
    base: Path = Path("data")
    logs: Path = Path("logs")
    reports: Path = Path("reports")

    def parquet_dir(self, exchange_id: str, timeframe: str, symbol_safe: str) -> Path:
        return self.base / "parquet" / exchange_id / timeframe / symbol_safe

    def state_dir(self, exchange_id: str, timeframe: str) -> Path:
        return self.base / "state" / exchange_id / timeframe


def setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sm_backtest.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)


# -----------------------------
# Utils
# -----------------------------
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_ms(ts: dt.datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)


def parse_iso_date_to_ms(s: str) -> int:
    # Accepts "YYYY-MM-DD" or full ISO. Assumes UTC if naive.
    s = s.strip()
    if len(s) == 10:
        ts = dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)
    else:
        ts = dt.datetime.fromisoformat(s)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
    return to_ms(ts)


def timeframe_to_ms(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86_400_000
    if tf.endswith("w"):
        return int(tf[:-1]) * 7 * 86_400_000
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def sanitize_symbol(symbol: str) -> str:
    return (
        symbol.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


# -----------------------------
# Data containers
# -----------------------------
@dataclass(frozen=True)
class MarketSeries:
    symbol: str
    ts_ms: np.ndarray  # int64, sorted, unique
    open: np.ndarray   # float64
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


@dataclass(frozen=True)
class MarketData:
    exchange_id: str
    timeframe: str
    symbols: List[str]
    ts_ms: np.ndarray            # (T,)
    open: np.ndarray             # (T, N)
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @property
    def T(self) -> int:
        return int(self.ts_ms.shape[0])

    @property
    def N(self) -> int:
        return int(self.open.shape[1])

    def bar_ms(self) -> int:
        if self.T < 2:
            return timeframe_to_ms(self.timeframe)
        diffs = np.diff(self.ts_ms.astype(np.int64))
        # robust median
        return int(np.median(diffs))


# -----------------------------
# Loading OHLCV from Parquet
# -----------------------------
def dataset_time_bounds(paths: Paths, exchange_id: str, timeframe: str, symbol: str) -> Tuple[int, int]:
    base_dir = paths.parquet_dir(exchange_id, timeframe, sanitize_symbol(symbol))
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing dataset for {symbol} at {base_dir}")
    lf = pl.scan_parquet(str(base_dir / "**" / "*.parquet")).select(
        [
            pl.col("timestamp_ms").cast(pl.Int64).min().alias("min_ts"),
            pl.col("timestamp_ms").cast(pl.Int64).max().alias("max_ts"),
        ]
    )
    df = collect_streaming(lf)
    return int(df["min_ts"][0]), int(df["max_ts"][0])


def load_ohlcv_symbol(
    paths: Paths,
    exchange_id: str,
    timeframe: str,
    symbol: str,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> MarketSeries:
    logger = logging.getLogger("load")

    symbol_safe = sanitize_symbol(symbol)
    base_dir = paths.parquet_dir(exchange_id, timeframe, symbol_safe)
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing OHLCV dataset for {symbol}: {base_dir}")

    lf = pl.scan_parquet(str(base_dir / "**" / "*.parquet")).select(
        [
            pl.col("timestamp_ms").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    )

    if start_ms is not None:
        lf = lf.filter(pl.col("timestamp_ms") >= int(start_ms))
    if end_ms is not None:
        lf = lf.filter(pl.col("timestamp_ms") < int(end_ms))

    lf = lf.sort("timestamp_ms")
    df = collect_streaming(lf)
    if df.is_empty():
        raise RuntimeError(f"Empty OHLCV after filtering for {symbol} at {base_dir}")

    # Basic integrity checks
    ts = df["timestamp_ms"].to_numpy()
    if ts.ndim != 1:
        raise RuntimeError("timestamp_ms invalid shape")
    if not np.all(ts[1:] > ts[:-1]):
        raise RuntimeError(f"{symbol}: timestamps not strictly increasing (duplicates/out-of-order)")

    # Convert to numpy float64
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()

    # NaN check
    for name, arr in [("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)]:
        if np.any(~np.isfinite(arr)):
            logger.warning("%s: non-finite values detected in %s", symbol, name)

    return MarketSeries(
        symbol=symbol,
        ts_ms=ts.astype(np.int64, copy=False),
        open=o.astype(np.float64, copy=False),
        high=h.astype(np.float64, copy=False),
        low=l.astype(np.float64, copy=False),
        close=c.astype(np.float64, copy=False),
        volume=v.astype(np.float64, copy=False),
    )


def align_on_intersection(series_list: Sequence[MarketSeries]) -> MarketData:
    if not series_list:
        raise ValueError("No series to align")

    exchange_id = "unknown"
    timeframe = "unknown"

    # intersection of timestamps
    ts_all = [s.ts_ms for s in series_list]
    ts_common = reduce(np.intersect1d, ts_all)
    if ts_common.size < 10:
        raise RuntimeError(f"Intersection too small: {ts_common.size} rows. Check datasets/time ranges.")

    # materialize matrices
    symbols = [s.symbol for s in series_list]
    T = int(ts_common.size)
    N = int(len(series_list))

    open_m = np.empty((T, N), dtype=np.float64)
    high_m = np.empty((T, N), dtype=np.float64)
    low_m = np.empty((T, N), dtype=np.float64)
    close_m = np.empty((T, N), dtype=np.float64)
    vol_m = np.empty((T, N), dtype=np.float64)

    for j, s in enumerate(series_list):
        idx = np.searchsorted(s.ts_ms, ts_common)
        if idx.size != T:
            raise RuntimeError("searchsorted mismatch")
        if np.any(idx < 0) or np.any(idx >= s.ts_ms.size) or not np.all(s.ts_ms[idx] == ts_common):
            raise RuntimeError(f"{s.symbol}: timestamp alignment failed (missing bars?)")
        open_m[:, j] = s.open[idx]
        high_m[:, j] = s.high[idx]
        low_m[:, j] = s.low[idx]
        close_m[:, j] = s.close[idx]
        vol_m[:, j] = s.volume[idx]

    # We don't store exchange/timeframe here; caller sets later.
    return MarketData(
        exchange_id=exchange_id,
        timeframe=timeframe,
        symbols=symbols,
        ts_ms=ts_common.astype(np.int64, copy=False),
        open=open_m,
        high=high_m,
        low=low_m,
        close=close_m,
        volume=vol_m,
    )


def discover_symbols_from_state(paths: Paths, exchange_id: str, timeframe: str) -> List[str]:
    st_dir = paths.state_dir(exchange_id, timeframe)
    if not st_dir.exists():
        return []
    symbols: List[str] = []
    for p in sorted(st_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            s = obj.get("symbol")
            if isinstance(s, str) and s.strip():
                symbols.append(s.strip())
        except Exception:
            continue
    # stable unique order
    out: List[str] = []
    seen = set()
    for s in symbols:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# -----------------------------
# Funding loader (optional)
# -----------------------------
def load_funding_for_symbols(
    paths: Paths,
    exchange_id: str,
    symbols: Sequence[str],
    ts_common: np.ndarray,
) -> Optional[np.ndarray]:
    logger = logging.getLogger("funding")
    T = int(ts_common.size)
    N = int(len(symbols))
    funding = np.zeros((T, N), dtype=np.float64)

    # derive candle grid size from ts_common (e.g. 1h -> 3600000)
    if T >= 2:
        bar_ms = int(np.median(np.diff(ts_common.astype(np.int64))))
        if bar_ms <= 0:
            bar_ms = 3_600_000
    else:
        bar_ms = 3_600_000

    any_found = False
    for j, sym in enumerate(symbols):
        base_dir = paths.parquet_dir(exchange_id, "funding", sanitize_symbol(sym))
        if not base_dir.exists():
            continue
        any_found = True

        lf = pl.scan_parquet(str(base_dir / "**" / "*.parquet")).select(
            [
                pl.col("timestamp_ms").cast(pl.Int64).alias("timestamp_ms"),
                pl.col("funding_rate").cast(pl.Float64).alias("funding_rate"),
            ]
        ).sort("timestamp_ms")

        df = collect_streaming(lf)
        if df.is_empty():
            continue

        f_ts = df["timestamp_ms"].to_numpy().astype(np.int64, copy=False)
        f_rt = df["funding_rate"].to_numpy().astype(np.float64, copy=False)

        # Align funding timestamps to the candle open grid (fix ms offsets like ...0007)
        f_ts_aligned = (f_ts // bar_ms) * bar_ms

        idx = np.searchsorted(ts_common, f_ts_aligned)
        valid = idx < T

        ts_match = np.zeros_like(valid, dtype=bool)
        if np.any(valid):
            ts_match[valid] = (ts_common[idx[valid]] == f_ts_aligned[valid])

        mask = valid & ts_match & np.isfinite(f_rt)

        if np.any(mask):
            funding[idx[mask], j] = f_rt[mask]

        logger.info("%s: funding events matched=%d/%d (grid=%dms)", sym, int(mask.sum()), int(f_ts.size), bar_ms)

    if not any_found:
        logger.warning("No funding datasets found under data/parquet/%s/funding/...", exchange_id)
        return None
    return funding


# -----------------------------
# Strategy primitives
# -----------------------------
def rolling_mean_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=True)
    if window > x.size:
        out = np.full_like(x, np.nan, dtype=np.float64)
        return out
    x = x.astype(np.float64, copy=False)
    c = np.cumsum(np.insert(x, 0, 0.0))
    rm = (c[window:] - c[:-window]) / float(window)
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    out[window - 1 :] = rm
    return out


@dataclass(frozen=True)
class MovingAverageCrossStrategy:
    fast: int = 48
    slow: int = 240
    leverage: float = 1.0
    long_only: bool = False

    def positions(self, md: MarketData) -> np.ndarray:
        """
        Signal computed on close[t] -> executed at open[t+1].
        We return positions aligned to open timestamps (md.ts_ms).
        """
        T, N = md.T, md.N
        pos = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            c = md.close[:, j]
            f = rolling_mean_1d(c, self.fast)
            s = rolling_mean_1d(c, self.slow)
            sig_close = np.where(np.isfinite(f) & np.isfinite(s) & (f > s), 1.0, -1.0)
            if self.long_only:
                sig_close = np.where(sig_close > 0, 1.0, 0.0)

            # shift for next-open execution
            # pos[0] cannot trade (no prior close), keep 0
            pos[1:, j] = sig_close[:-1] * float(self.leverage)

            # warmup: until slow MA valid, keep 0
            warm = max(self.fast, self.slow)
            pos[: max(1, warm), j] = 0.0
        return pos


# -----------------------------
# Backtest core
# -----------------------------
@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 1.0

    # Costs: use bps inputs in CLI, internally use rates (fraction)
    fee_rate: float = 0.0004          # 4 bps (typical taker-ish, tune per exchange)
    slippage_rate: float = 0.00005    # 0.5 bps

    # Risk constraints on position matrix (units = "x equity" notional)
    max_gross_leverage: float = 1.0
    max_abs_position: float = 1.0     # per asset

    # If True, scale positions to satisfy max_gross_leverage (keeps relative weights)
    scale_to_gross: bool = True


@dataclass(frozen=True)
class BacktestResult:
    ts_ms: np.ndarray
    equity: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    metrics: Dict[str, Any]


def apply_position_constraints(pos: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    pos = pos.astype(np.float64, copy=True)

    # per-asset cap
    if cfg.max_abs_position > 0:
        pos = np.clip(pos, -cfg.max_abs_position, cfg.max_abs_position)

    # gross leverage cap
    if cfg.max_gross_leverage > 0 and cfg.scale_to_gross:
        gross = np.sum(np.abs(pos), axis=1)
        scale = np.ones_like(gross, dtype=np.float64)
        mask = gross > cfg.max_gross_leverage
        scale[mask] = cfg.max_gross_leverage / gross[mask]
        pos = pos * scale[:, None]
    return pos


def compute_metrics(ts_ms: np.ndarray, equity: np.ndarray, returns: np.ndarray, bar_ms: int) -> Dict[str, Any]:
    # returns are per "bar", equity aligned accordingly
    eps = 1e-12

    # duration
    total_seconds = (ts_ms[-1] - ts_ms[0]) / 1000.0
    years = max(total_seconds / (365.25 * 24 * 3600), eps)

    # annualization
    bars_per_year = (365.25 * 24 * 3600) / max(bar_ms / 1000.0, eps)

    r = returns.astype(np.float64, copy=False)
    mu = float(np.nanmean(r))
    sd = float(np.nanstd(r, ddof=1)) if r.size > 2 else float("nan")
    vol = sd * math.sqrt(bars_per_year) if math.isfinite(sd) and sd > 0 else float("nan")
    sharpe = (mu / sd) * math.sqrt(bars_per_year) if math.isfinite(sd) and sd > 0 else float("nan")

    downside = r[r < 0]
    dd_sd = float(np.nanstd(downside, ddof=1)) if downside.size > 2 else float("nan")
    sortino = (mu / dd_sd) * math.sqrt(bars_per_year) if math.isfinite(dd_sd) and dd_sd > 0 else float("nan")

    eq0 = float(equity[0])
    eq1 = float(equity[-1])
    cagr = (eq1 / max(eq0, eps)) ** (1.0 / years) - 1.0

    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, eps) - 1.0
    max_dd = float(np.min(dd))
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    # profit factor on bar returns (not trade-level, but still useful)
    pos_sum = float(np.sum(r[r > 0]))
    neg_sum = float(np.sum(r[r < 0]))
    profit_factor = (pos_sum / abs(neg_sum)) if neg_sum < 0 else float("inf")

    # monthly return stats (robustness proxy)
    try:
        df = pl.DataFrame({"ts": ts_ms.astype(np.int64), "equity": equity.astype(np.float64)}).with_columns(
            pl.from_epoch(pl.col("ts") / 1000, time_unit="s").alias("dt")
        ).with_columns(
            [pl.col("dt").dt.year().alias("year"), pl.col("dt").dt.month().alias("month")]
        )
        m = (
            df.group_by(["year", "month"])
            .agg([pl.col("equity").first().alias("first"), pl.col("equity").last().alias("last")])
            .with_columns((pl.col("last") / pl.col("first") - 1.0).alias("ret"))
            .sort(["year", "month"])
        )
        mrets = m["ret"].to_numpy()
        pct_pos_months = float(np.mean(mrets > 0)) if mrets.size else float("nan")
        worst_month = float(np.min(mrets)) if mrets.size else float("nan")
        best_month = float(np.max(mrets)) if mrets.size else float("nan")
    except Exception:
        pct_pos_months = float("nan")
        worst_month = float("nan")
        best_month = float("nan")

    return {
        "equity_start": eq0,
        "equity_end": eq1,
        "years": years,
        "cagr": cagr,
        "vol_annual": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "profit_factor_bars": profit_factor,
        "mean_bar_return": mu,
        "bars_per_year": bars_per_year,
        "pct_positive_months": pct_pos_months,
        "best_month": best_month,
        "worst_month": worst_month,
    }


def run_backtest(
    md: MarketData,
    positions: np.ndarray,
    cfg: BacktestConfig,
    *,
    funding_rate: Optional[np.ndarray] = None,  # (T, N), nonzero only at funding timestamps
) -> BacktestResult:
    logger = logging.getLogger("bt")

    T, N = md.T, md.N
    if positions.shape != (T, N):
        raise ValueError(f"positions shape {positions.shape} != (T,N)=({T},{N})")

    pos = apply_position_constraints(positions, cfg)

    # Price returns per bar:
    # t=0..T-2 -> open[t] -> open[t+1]
    # last bar t=T-1 -> open[T-1] -> close[T-1]
    open_px = md.open
    close_px = md.close

    r = np.empty((T, N), dtype=np.float64)
    r[:-1, :] = (open_px[1:, :] / open_px[:-1, :]) - 1.0
    r[-1, :] = (close_px[-1, :] / open_px[-1, :]) - 1.0

    # Turnover costs at each open[t]
    pos_prev = np.zeros((T, N), dtype=np.float64)
    pos_prev[1:, :] = pos[:-1, :]
    dpos = pos - pos_prev
    turnover = np.sum(np.abs(dpos), axis=1)  # (T,)
    costs = (cfg.fee_rate + cfg.slippage_rate) * turnover  # fraction of equity

    # Funding PnL at timestamps where funding is nonzero
    if funding_rate is not None:
        if funding_rate.shape != (T, N):
            raise ValueError(f"funding_rate shape {funding_rate.shape} != (T,N)=({T},{N})")
        # Funding applies to the position held *into* the funding timestamp
        funding_pnl = -np.sum(pos_prev * funding_rate, axis=1)
    else:
        funding_pnl = np.zeros(T, dtype=np.float64)

    # Portfolio bar returns
    ret = np.sum(pos * r, axis=1) + funding_pnl - costs  # (T,)

    # Close-out cost after last close (apply multiplicatively as final step)
    closeout_cost = (cfg.fee_rate + cfg.slippage_rate) * float(np.sum(np.abs(pos[-1, :])))
    closeout_step = 1.0 - closeout_cost

    # Build equity curve safely (handle bankrupt / negative step)
    gross_steps = 1.0 + ret
    equity = np.empty(T + 2, dtype=np.float64)
    equity[0] = float(cfg.initial_equity)

    bankrupt_at = None
    for t in range(T):
        step = gross_steps[t]
        if not np.isfinite(step):
            step = 0.0
        if step <= 0.0:
            equity[t + 1] = 0.0
            bankrupt_at = t
            equity[t + 2 :] = 0.0
            break
        equity[t + 1] = equity[t] * step

    if bankrupt_at is None:
        # apply closeout step
        if closeout_step <= 0.0:
            equity[-1] = 0.0
        else:
            equity[-1] = equity[-2] * closeout_step

    # timestamps for equity points:
    # start at open[0], then after each bar we use open[t+1] (or last close)
    bar_ms = md.bar_ms()
    ts_eq = np.empty(T + 2, dtype=np.int64)
    ts_eq[0] = md.ts_ms[0]
    ts_eq[1 : T + 1] = md.ts_ms
    ts_eq[-1] = int(md.ts_ms[-1] + bar_ms)  # approximate "last close time"

    # metrics (use equity including closeout; returns include bar returns only, closeout as separate step)
    # We'll append the closeout as a final return for metrics consistency.
    ret_for_metrics = np.concatenate([ret, np.array([closeout_step - 1.0], dtype=np.float64)])
    bar_ms_for_metrics = bar_ms
    metrics = compute_metrics(ts_eq, equity, ret_for_metrics, bar_ms_for_metrics)

    # diagnostics
    metrics["turnover_mean"] = float(np.mean(turnover))
    metrics["turnover_p95"] = float(np.percentile(turnover, 95))
    metrics["closeout_cost_frac"] = float(closeout_cost)
    if bankrupt_at is not None:
        metrics["bankrupt_at_bar"] = int(bankrupt_at)

    logger.info(
        "BT done | equity_end=%.4f | cagr=%.3f | sharpe=%s | maxDD=%.3f | closeout=%.5f",
        float(equity[-1]),
        float(metrics.get("cagr", float("nan"))),
        "nan" if not math.isfinite(float(metrics.get("sharpe", float("nan")))) else f"{metrics.get('sharpe'):.2f}",
        float(metrics.get("max_drawdown", float("nan"))),
        float(closeout_cost),
    )

    return BacktestResult(
        ts_ms=ts_eq,
        equity=equity,
        returns=ret_for_metrics,
        positions=pos,
        metrics=metrics,
    )


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sm_backtest.py", description="Strategy Miner - Backtest Core")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a sample strategy on local Parquet data.")
    run.add_argument("--exchange", default="binanceusdm")
    run.add_argument("--timeframe", default="1h")
    run.add_argument("--symbols", default="", help='Comma list, e.g. "BTC/USDT:USDT,ETH/USDT:USDT". If empty: discover from state dir.')
    run.add_argument("--top-n", type=int, default=2, help="If symbols empty: use first top-n discovered (default: 2)")
    run.add_argument("--years", type=float, default=0.0, help="Use last N years from available data (0=all)")
    run.add_argument("--start", default="", help="ISO date/time (UTC assumed if no tz). Example: 2023-01-01")
    run.add_argument("--end", default="", help="ISO date/time (UTC assumed if no tz). Example: 2025-01-01")
    run.add_argument("--funding", action="store_true", help="Load funding datasets if present and include in PnL.")

    run.add_argument("--strategy", default="ma", choices=["ma"], help="Strategy type (default: ma)")
    run.add_argument("--fast", type=int, default=48)
    run.add_argument("--slow", type=int, default=240)
    run.add_argument("--leverage", type=float, default=1.0)
    run.add_argument("--long-only", action="store_true")

    run.add_argument("--fee-bps", type=float, default=4.0)
    run.add_argument("--slippage-bps", type=float, default=0.5)
    run.add_argument("--max-gross", type=float, default=1.0)
    run.add_argument("--max-pos", type=float, default=1.0)

    run.add_argument("--out", default="", help="Optional: write metrics JSON to file (in ./reports).")
    run.add_argument("--quiet", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    verbose = not bool(getattr(args, "quiet", False))
    paths = Paths()
    setup_logging(paths.logs, verbose=verbose)

    logger = logging.getLogger("main")

    exchange_id = args.exchange
    timeframe = args.timeframe

    if args.cmd == "run":
        if args.symbols.strip():
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            discovered = discover_symbols_from_state(paths, exchange_id, timeframe)
            if not discovered:
                raise RuntimeError(
                    f"No symbols provided and none discovered in {paths.state_dir(exchange_id, timeframe)}. "
                    f"Run sm_data.py download first."
                )
            symbols = discovered[: int(args.top_n)]

        # determine time window
        start_ms = parse_iso_date_to_ms(args.start) if args.start.strip() else None
        end_ms = parse_iso_date_to_ms(args.end) if args.end.strip() else None

        if args.years and args.years > 0 and (start_ms is None and end_ms is None):
            # choose last N years based on available bounds across symbols
            bounds = [dataset_time_bounds(paths, exchange_id, timeframe, s) for s in symbols]
            end_ms_eff = min(mx for _, mx in bounds)
            start_ms_eff = int(end_ms_eff - (args.years * 365.25 * 24 * 3600 * 1000))
            start_ms = start_ms_eff
            end_ms = end_ms_eff
            logger.info("Window from available data: start=%d end=%d (years=%.2f)", start_ms, end_ms, float(args.years))

        # load series
        series_list = [
            load_ohlcv_symbol(paths, exchange_id, timeframe, s, start_ms=start_ms, end_ms=end_ms)
            for s in symbols
        ]
        md0 = align_on_intersection(series_list)
        md = MarketData(
            exchange_id=exchange_id,
            timeframe=timeframe,
            symbols=md0.symbols,
            ts_ms=md0.ts_ms,
            open=md0.open,
            high=md0.high,
            low=md0.low,
            close=md0.close,
            volume=md0.volume,
        )
        logger.info("Loaded aligned data: T=%d bars, N=%d symbols=%s", md.T, md.N, ",".join(md.symbols))

        # strategy
        if args.strategy == "ma":
            strat = MovingAverageCrossStrategy(
                fast=int(args.fast),
                slow=int(args.slow),
                leverage=float(args.leverage),
                long_only=bool(args.long_only),
            )
            pos = strat.positions(md)
        else:
            raise ValueError("Unknown strategy")

        cfg = BacktestConfig(
            initial_equity=1.0,
            fee_rate=float(args.fee_bps) / 10_000.0,
            slippage_rate=float(args.slippage_bps) / 10_000.0,
            max_gross_leverage=float(args.max_gross),
            max_abs_position=float(args.max_pos),
        )

        funding = None
        if bool(args.funding):
            funding = load_funding_for_symbols(paths, exchange_id, md.symbols, md.ts_ms)

        res = run_backtest(md, pos, cfg, funding_rate=funding)

        # print metrics
        metrics = dict(res.metrics)
        metrics["exchange"] = exchange_id
        metrics["timeframe"] = timeframe
        metrics["symbols"] = md.symbols
        metrics["bars"] = md.T
        print(json.dumps(metrics, indent=2, sort_keys=True))

        # optional write
        if args.out.strip():
            paths.reports.mkdir(parents=True, exist_ok=True)
            out_path = paths.reports / args.out.strip()
            out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
            logger.info("Wrote report: %s", out_path)

        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
