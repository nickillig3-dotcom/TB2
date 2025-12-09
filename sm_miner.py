#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm_miner.py — Strategy Miner (parameter search + walk-forward) for perp futures datasets

- Offline: uses local Parquet from sm_data.py
- Reuses backtest core from sm_backtest.py
- Walk-forward evaluation (rolling windows)
- Writes results as Parquet (no CSV)

Requires:
  pip install -U numpy polars pyarrow
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

import sm_backtest as bt


# -----------------------------
# Logging
# -----------------------------
def setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sm_miner.log"

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
# WF / Search config
# -----------------------------
@dataclass(frozen=True)
class WfConfig:
    train_months: float = 24.0
    test_months: float = 6.0
    step_months: float = 6.0
    mode: str = "rolling"  # rolling | expanding


@dataclass(frozen=True)
class SearchConfig:
    strategy: str = "ma"
    mode: str = "grid"  # grid | random
    samples: int = 200
    seed: int = 42
    leverage: float = 1.0
    long_only: bool = False

    fast_values: Tuple[int, ...] = (24, 48, 72)
    slow_values: Tuple[int, ...] = (120, 240, 360)

    top_k: int = 20


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def month_bars(months: float, bar_ms: int) -> int:
    # 365.25/12 days per month average
    seconds = months * (365.25 / 12.0) * 24.0 * 3600.0
    return max(1, int(round(seconds / max(bar_ms / 1000.0, 1e-9))))


def make_folds(T: int, wf: WfConfig, bar_ms: int) -> List[Tuple[int, int]]:
    """
    Returns list of (test_start, test_end) indices.
    We always compute signals using full history, but evaluate each test slice from flat.
    """
    train_b = month_bars(wf.train_months, bar_ms)
    test_b = month_bars(wf.test_months, bar_ms)
    step_b = month_bars(wf.step_months, bar_ms)

    folds: List[Tuple[int, int]] = []
    if wf.mode not in ("rolling", "expanding"):
        raise ValueError("wf.mode must be rolling or expanding")

    start = 0
    while True:
        if wf.mode == "rolling":
            train_start = start
        else:
            train_start = 0

        train_end = start + train_b
        test_start = train_end
        test_end = test_start + test_b

        if test_end > T:
            break

        folds.append((test_start, test_end))
        start += step_b

        # avoid infinite loops
        if step_b <= 0:
            break

    return folds


def rolling_mean_matrix(X: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorized rolling mean over axis=0 (time). Output shape (T,N) with NaNs for first window-1 rows.
    """
    X = X.astype(np.float64, copy=False)
    T, N = X.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if window <= 1:
        return X.copy()
    if window > T:
        return out
    cs = np.cumsum(X, axis=0, dtype=np.float64)
    cs = np.vstack([np.zeros((1, N), dtype=np.float64), cs])
    rm = (cs[window:] - cs[:-window]) / float(window)  # (T-window+1, N)
    out[window - 1 :] = rm
    return out


def build_ma_positions(
    close: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    *,
    leverage: float,
    long_only: bool,
    warmup: int,
) -> np.ndarray:
    """
    close/ma_* shape (T,N). Returns pos shape (T,N) aligned to open timestamps (next-open execution).
    Signal computed on close[t] -> executed at open[t+1].
    """
    T, N = close.shape
    pos = np.zeros((T, N), dtype=np.float64)

    valid = np.isfinite(ma_fast) & np.isfinite(ma_slow)
    sig_close = np.where(valid & (ma_fast > ma_slow), 1.0, -1.0)
    if long_only:
        sig_close = np.where(sig_close > 0.0, 1.0, 0.0)

    pos[1:, :] = sig_close[:-1, :] * float(leverage)

    # warmup: until both MAs valid -> 0
    warm = max(1, int(warmup))
    pos[:warm, :] = 0.0
    return pos


def generate_candidates(cfg: SearchConfig) -> List[Tuple[int, int]]:
    fast = [int(x) for x in cfg.fast_values if int(x) > 0]
    slow = [int(x) for x in cfg.slow_values if int(x) > 0]
    fast = sorted(set(fast))
    slow = sorted(set(slow))

    pairs = [(f, s) for f in fast for s in slow if f < s]
    if not pairs:
        raise ValueError("No valid (fast,slow) pairs. Ensure fast < slow and both lists non-empty.")

    if cfg.mode == "grid":
        return pairs

    if cfg.mode == "random":
        rng = random.Random(int(cfg.seed))
        # sample unique pairs without replacement
        k = min(int(cfg.samples), len(pairs))
        rng.shuffle(pairs)
        return pairs[:k]

    raise ValueError("cfg.mode must be grid or random")


def summarize_folds(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def ensure_reports_dir(paths: bt.Paths) -> Path:
    paths.reports.mkdir(parents=True, exist_ok=True)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    out_dir = paths.reports / "miner_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sm_miner.py", description="Strategy Miner (walk-forward parameter search)")
    sub = p.add_subparsers(dest="cmd", required=True)

    mine = sub.add_parser("mine", help="Run walk-forward search on local Parquet data.")
    mine.add_argument("--exchange", default="binanceusdm")
    mine.add_argument("--timeframe", default="1h")
    mine.add_argument("--symbols", default="", help='Comma list. If empty: discover from state dir.')
    mine.add_argument("--top-n", type=int, default=10, help="If symbols empty: use first top-n discovered.")
    mine.add_argument("--years", type=float, default=0.0, help="Use last N years from available data (0=all).")

    mine.add_argument("--strategy", default="ma", choices=["ma"])
    mine.add_argument("--mode", default="grid", choices=["grid", "random"])
    mine.add_argument("--samples", type=int, default=200)
    mine.add_argument("--seed", type=int, default=42)

    mine.add_argument("--fast-min", type=int, default=12)
    mine.add_argument("--fast-max", type=int, default=96)
    mine.add_argument("--fast-step", type=int, default=12)
    mine.add_argument("--slow-min", type=int, default=120)
    mine.add_argument("--slow-max", type=int, default=480)
    mine.add_argument("--slow-step", type=int, default=60)

    mine.add_argument("--leverage", type=float, default=1.0)
    mine.add_argument("--long-only", action="store_true")

    mine.add_argument("--train-months", type=float, default=24.0)
    mine.add_argument("--test-months", type=float, default=6.0)
    mine.add_argument("--step-months", type=float, default=6.0)
    mine.add_argument("--wf-mode", default="rolling", choices=["rolling", "expanding"])

    mine.add_argument("--fee-bps", type=float, default=4.0)
    mine.add_argument("--slippage-bps", type=float, default=0.5)
    mine.add_argument("--max-gross", type=float, default=1.0)
    mine.add_argument("--max-pos", type=float, default=1.0)

    mine.add_argument("--funding", action="store_true", help="Include funding if datasets exist.")
    mine.add_argument("--top-k", type=int, default=20)
    mine.add_argument("--quiet", action="store_true")

    return p.parse_args()


def frange(min_v: int, max_v: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if max_v < min_v:
        return []
    return list(range(int(min_v), int(max_v) + 1, int(step)))


def main() -> int:
    args = parse_args()
    verbose = not bool(args.quiet)

    paths = bt.Paths()
    setup_logging(paths.logs, verbose=verbose)
    logger = logging.getLogger("miner")

    if args.cmd != "mine":
        return 2

    exchange_id = args.exchange
    timeframe = args.timeframe

    # symbols
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        discovered = bt.discover_symbols_from_state(paths, exchange_id, timeframe)
        if not discovered:
            raise RuntimeError(
                f"No symbols provided and none discovered in {paths.state_dir(exchange_id, timeframe)}. "
                f"Run sm_data.py download first."
            )
        symbols = discovered[: int(args.top_n)]

    # determine time window (last N years from available bounds across symbols)
    start_ms = None
    end_ms = None
    if args.years and float(args.years) > 0:
        bounds = [bt.dataset_time_bounds(paths, exchange_id, timeframe, s) for s in symbols]
        end_ms_eff = min(mx for _, mx in bounds)
        start_ms_eff = int(end_ms_eff - (float(args.years) * 365.25 * 24 * 3600 * 1000))
        start_ms = start_ms_eff
        end_ms = end_ms_eff
        logger.info("Window from available data: start=%d end=%d (years=%.2f)", start_ms, end_ms, float(args.years))

    # load + align
    series_list = [
        bt.load_ohlcv_symbol(paths, exchange_id, timeframe, s, start_ms=start_ms, end_ms=end_ms)
        for s in symbols
    ]
    md0 = bt.align_on_intersection(series_list)
    md = bt.MarketData(
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

    bar_ms = md.bar_ms()

    wf = WfConfig(
        train_months=float(args.train_months),
        test_months=float(args.test_months),
        step_months=float(args.step_months),
        mode=str(args.wf_mode),
    )
    folds = make_folds(md.T, wf, bar_ms)
    if len(folds) == 0:
        raise RuntimeError(
            f"Not enough data for walk-forward. T={md.T} bars; "
            f"train={wf.train_months}m test={wf.test_months}m step={wf.step_months}m."
        )
    logger.info("Walk-forward folds=%d | mode=%s | train=%.2fm test=%.2fm step=%.2fm",
                len(folds), wf.mode, wf.train_months, wf.test_months, wf.step_months)

    # search space
    fast_vals = frange(args.fast_min, args.fast_max, args.fast_step)
    slow_vals = frange(args.slow_min, args.slow_max, args.slow_step)
    scfg = SearchConfig(
        strategy=str(args.strategy),
        mode=str(args.mode),
        samples=int(args.samples),
        seed=int(args.seed),
        leverage=float(args.leverage),
        long_only=bool(args.long_only),
        fast_values=tuple(fast_vals),
        slow_values=tuple(slow_vals),
        top_k=int(args.top_k),
    )
    candidates = generate_candidates(scfg)
    logger.info("Candidates=%d | search_mode=%s | strategy=%s", len(candidates), scfg.mode, scfg.strategy)

    # precompute MA cache for unique windows
    uniq_windows = sorted(set(list(scfg.fast_values) + list(scfg.slow_values)))
    ma_cache: Dict[int, np.ndarray] = {}
    logger.info("Precomputing rolling means for %d unique windows...", len(uniq_windows))
    for w in uniq_windows:
        ma_cache[w] = rolling_mean_matrix(md.close, int(w))

    # optional funding
    funding = None
    if bool(args.funding):
        funding = bt.load_funding_for_symbols(paths, exchange_id, md.symbols, md.ts_ms)

    # backtest config
    btcfg = bt.BacktestConfig(
        initial_equity=1.0,
        fee_rate=float(args.fee_bps) / 10_000.0,
        slippage_rate=float(args.slippage_bps) / 10_000.0,
        max_gross_leverage=float(args.max_gross),
        max_abs_position=float(args.max_pos),
        scale_to_gross=True,
    )

    out_dir = ensure_reports_dir(paths)

    rows: List[Dict[str, Any]] = []
    for i, (fast, slow) in enumerate(candidates, 1):
        warmup = max(int(fast), int(slow))
        pos = build_ma_positions(
            md.close,
            ma_cache[int(fast)],
            ma_cache[int(slow)],
            leverage=scfg.leverage,
            long_only=scfg.long_only,
            warmup=warmup,
        )

        fold_sharpe: List[float] = []
        fold_cagr: List[float] = []
        fold_mdd: List[float] = []
        fold_end: List[float] = []
        fold_turn: List[float] = []

        for (ts0, ts1) in folds:
            md_slice = bt.MarketData(
                exchange_id=md.exchange_id,
                timeframe=md.timeframe,
                symbols=md.symbols,
                ts_ms=md.ts_ms[ts0:ts1],
                open=md.open[ts0:ts1],
                high=md.high[ts0:ts1],
                low=md.low[ts0:ts1],
                close=md.close[ts0:ts1],
                volume=md.volume[ts0:ts1],
            )
            pos_slice = pos[ts0:ts1, :]
            funding_slice = funding[ts0:ts1, :] if funding is not None else None

            res = bt.run_backtest(md_slice, pos_slice, btcfg, funding_rate=funding_slice)
            m = res.metrics
            fold_sharpe.append(float(m.get("sharpe", float("nan"))))
            fold_cagr.append(float(m.get("cagr", float("nan"))))
            fold_mdd.append(float(m.get("max_drawdown", float("nan"))))
            fold_end.append(float(m.get("equity_end", float("nan"))))
            fold_turn.append(float(m.get("turnover_mean", float("nan"))))

        sh = summarize_folds(fold_sharpe)
        cg = summarize_folds(fold_cagr)
        dd = summarize_folds(fold_mdd)
        ed = summarize_folds(fold_end)
        tn = summarize_folds(fold_turn)

        pos_folds = float(np.mean(np.array(fold_end, dtype=np.float64) > 1.0)) if fold_end else float("nan")
        worst_dd = dd["min"]  # most negative
        # simple, transparent score (not magical): reward median CAGR + median Sharpe, penalize worst drawdown
        score = cg["median"] + 0.25 * sh["median"] - 0.5 * abs(worst_dd) if math.isfinite(worst_dd) else float("nan")

        rows.append(
            {
                "fast": int(fast),
                "slow": int(slow),
                "leverage": float(scfg.leverage),
                "long_only": bool(scfg.long_only),
                "folds": int(len(folds)),
                "fold_sharpe": fold_sharpe,
                "fold_cagr": fold_cagr,
                "fold_max_dd": fold_mdd,
                "fold_equity_end": fold_end,
                "fold_turnover_mean": fold_turn,
                "test_sharpe_mean": sh["mean"],
                "test_sharpe_median": sh["median"],
                "test_cagr_mean": cg["mean"],
                "test_cagr_median": cg["median"],
                "test_mdd_worst": dd["min"],
                "test_mdd_median": dd["median"],
                "test_equity_end_median": ed["median"],
                "test_pos_folds": pos_folds,
                "turnover_mean_median": tn["median"],
                "score": score,
            }
        )

        if i % max(1, len(candidates) // 10) == 0 or i == len(candidates):
            logger.info("Progress %d/%d", i, len(candidates))

    df = pl.DataFrame(rows)

    # Sort results
    df = df.sort(["score", "test_sharpe_median", "test_cagr_median"], descending=True)

    # Print top-k
    k = min(int(scfg.top_k), df.height)
    top = df.head(k).select(
        [
            "fast",
            "slow",
            "score",
            "test_sharpe_median",
            "test_cagr_median",
            "test_mdd_worst",
            "test_pos_folds",
            "turnover_mean_median",
        ]
    )
    print(top)

    # Write parquet + run metadata
    out_parquet = out_dir / f"results_{exchange_id}_{timeframe}_{scfg.strategy}.parquet"
    df.write_parquet(out_parquet, compression="zstd")

    meta = {
        "exchange": exchange_id,
        "timeframe": timeframe,
        "symbols": md.symbols,
        "T": md.T,
        "N": md.N,
        "wf": {
            "mode": wf.mode,
            "train_months": wf.train_months,
            "test_months": wf.test_months,
            "step_months": wf.step_months,
            "folds": len(folds),
        },
        "search": {
            "strategy": scfg.strategy,
            "mode": scfg.mode,
            "candidates": len(candidates),
            "seed": scfg.seed,
            "leverage": scfg.leverage,
            "long_only": scfg.long_only,
            "fast_values": list(scfg.fast_values),
            "slow_values": list(scfg.slow_values),
        },
        "costs": {
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slippage_bps),
            "max_gross": float(args.max_gross),
            "max_pos": float(args.max_pos),
            "funding": bool(args.funding),
        },
        "outputs": {"parquet": str(out_parquet)},
        "created_at_utc": utc_now().isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote results: %s", out_parquet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
