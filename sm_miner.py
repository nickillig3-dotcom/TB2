#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm_miner.py — Strategy Miner (walk-forward parameter search + optional holdout)

Upgrades:
- Start each WF test fold flat (no position carry-in from training)
- Optional final holdout window (last X months) for anti-overfit confirmation
- More robust scoring + explicit pass/fail filters (robustness-first)

Offline: uses local Parquet created by sm_data.py via sm_backtest loaders.

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
# Config
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


@dataclass(frozen=True)
class FilterConfig:
    # Train/WF filters
    min_pos_folds: float = 0.66          # 2/3 folds positive passes
    min_sharpe_median: float = 0.0
    min_equity_end_median: float = 1.0
    max_dd_worst: float = 0.35           # worst drawdown must be >= -0.35
    max_turnover_median: float = 0.10

    # Holdout filters (only used if holdout enabled)
    min_holdout_equity_end: float = 1.0
    max_holdout_dd: float = 0.35


# -----------------------------
# Utils
# -----------------------------
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def month_bars(months: float, bar_ms: int) -> int:
    seconds = months * (365.25 / 12.0) * 24.0 * 3600.0
    return max(1, int(round(seconds / max(bar_ms / 1000.0, 1e-9))))


def make_folds(T: int, wf: WfConfig, bar_ms: int) -> List[Tuple[int, int]]:
    train_b = month_bars(wf.train_months, bar_ms)
    test_b = month_bars(wf.test_months, bar_ms)
    step_b = month_bars(wf.step_months, bar_ms)

    if wf.mode not in ("rolling", "expanding"):
        raise ValueError("wf.mode must be rolling or expanding")
    if train_b <= 0 or test_b <= 0 or step_b <= 0:
        raise ValueError("train/test/step must be > 0")

    folds: List[Tuple[int, int]] = []
    start = 0
    while True:
        train_end = start + train_b
        test_start = train_end
        test_end = test_start + test_b
        if test_end > T:
            break
        folds.append((test_start, test_end))
        start += step_b
        if start < 0:
            break
    return folds


def rolling_mean_matrix(X: np.ndarray, window: int) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    T, N = X.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    if window <= 1:
        return X.copy()
    if window > T:
        return out
    cs = np.cumsum(X, axis=0, dtype=np.float64)
    cs = np.vstack([np.zeros((1, N), dtype=np.float64), cs])
    rm = (cs[window:] - cs[:-window]) / float(window)
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
    T, N = close.shape
    pos = np.zeros((T, N), dtype=np.float64)

    valid = np.isfinite(ma_fast) & np.isfinite(ma_slow)
    sig_close = np.where(valid & (ma_fast > ma_slow), 1.0, -1.0)
    if long_only:
        sig_close = np.where(sig_close > 0.0, 1.0, 0.0)

    pos[1:, :] = sig_close[:-1, :] * float(leverage)

    warm = max(1, int(warmup))
    pos[:warm, :] = 0.0
    return pos


def frange(min_v: int, max_v: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if max_v < min_v:
        return []
    return list(range(int(min_v), int(max_v) + 1, int(step)))


def generate_candidates(cfg: SearchConfig) -> List[Tuple[int, int]]:
    fast = sorted(set(int(x) for x in cfg.fast_values if int(x) > 0))
    slow = sorted(set(int(x) for x in cfg.slow_values if int(x) > 0))
    pairs = [(f, s) for f in fast for s in slow if f < s]
    if not pairs:
        raise ValueError("No valid (fast,slow) pairs. Ensure fast < slow and both lists non-empty.")

    if cfg.mode == "grid":
        return pairs
    if cfg.mode == "random":
        rng = random.Random(int(cfg.seed))
        rng.shuffle(pairs)
        return pairs[: min(int(cfg.samples), len(pairs))]
    raise ValueError("cfg.mode must be grid or random")


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=np.float64)
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


def slice_marketdata(md: bt.MarketData, i0: int, i1: int) -> bt.MarketData:
    return bt.MarketData(
        exchange_id=md.exchange_id,
        timeframe=md.timeframe,
        symbols=md.symbols,
        ts_ms=md.ts_ms[i0:i1],
        open=md.open[i0:i1],
        high=md.high[i0:i1],
        low=md.low[i0:i1],
        close=md.close[i0:i1],
        volume=md.volume[i0:i1],
    )


def score_train(
    *,
    eq_end_median: float,
    eq_end_min: float,
    sharpe_median: float,
    sharpe_min: float,
    dd_worst: float,
    pos_folds: float,
    turnover_median: float,
) -> float:
    ret_med = eq_end_median - 1.0
    ret_min = eq_end_min - 1.0
    dd_pen = abs(dd_worst)
    turn_pen = max(0.0, turnover_median)

    return float(
        6.0 * ret_med
        + 2.0 * (pos_folds - 0.5)
        + 0.15 * sharpe_median
        + 0.05 * sharpe_min
        - 0.50 * dd_pen
        - 0.05 * turn_pen
        + 1.0 * ret_min
    )


def passes_filters(
    *,
    fcfg: FilterConfig,
    pos_folds: float,
    sharpe_median: float,
    eq_end_median: float,
    dd_worst: float,
    turnover_median: float,
) -> bool:
    if not (math.isfinite(pos_folds) and math.isfinite(sharpe_median) and math.isfinite(eq_end_median)):
        return False
    if not math.isfinite(dd_worst) or not math.isfinite(turnover_median):
        return False
    if pos_folds < fcfg.min_pos_folds:
        return False
    if sharpe_median < fcfg.min_sharpe_median:
        return False
    if eq_end_median < fcfg.min_equity_end_median:
        return False
    if dd_worst < -abs(fcfg.max_dd_worst):
        return False
    if turnover_median > fcfg.max_turnover_median:
        return False
    return True


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

    mine.add_argument("--holdout-months", type=float, default=0.0, help="Final holdout window at the end (0=disable).")
    mine.add_argument("--holdout-top", type=int, default=30, help="Evaluate holdout only for top-N by train score.")

    mine.add_argument("--fee-bps", type=float, default=4.0)
    mine.add_argument("--slippage-bps", type=float, default=0.5)
    mine.add_argument("--max-gross", type=float, default=1.0)
    mine.add_argument("--max-pos", type=float, default=1.0)

    mine.add_argument("--funding", action="store_true", help="Include funding if datasets exist.")

    # Filters
    mine.add_argument("--min-pos-folds", type=float, default=0.66)
    mine.add_argument("--min-sharpe-median", type=float, default=0.0)
    mine.add_argument("--min-eq-median", type=float, default=1.0)
    mine.add_argument("--max-dd-worst", type=float, default=0.35)
    mine.add_argument("--max-turnover-median", type=float, default=0.10)
    mine.add_argument("--min-hold-eq", type=float, default=1.0)
    mine.add_argument("--max-hold-dd", type=float, default=0.35)

    mine.add_argument("--top-k", type=int, default=20)
    mine.add_argument("--require-pass", action="store_true", help="Only print strategies that pass filters.")
    mine.add_argument("--quiet", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    verbose = not bool(args.quiet)

    paths = bt.Paths()
    setup_logging(paths.logs, verbose=verbose)

    # Silence per-candidate backtest logs
    logging.getLogger("bt").setLevel(logging.WARNING)

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

    # last N years window
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

    # Holdout split
    holdout_months = float(args.holdout_months)
    holdout_start = md.T
    md_train = md
    md_hold: Optional[bt.MarketData] = None

    if holdout_months > 0:
        holdout_b = month_bars(holdout_months, bar_ms)
        if holdout_b >= md.T - 50:
            raise RuntimeError(f"Holdout too large: holdout_bars={holdout_b} for T={md.T}")
        holdout_start = md.T - holdout_b
        md_train = slice_marketdata(md, 0, holdout_start)
        md_hold = slice_marketdata(md, holdout_start, md.T)
        logger.info("Holdout enabled: months=%.2f | holdout_bars=%d | train_T=%d | hold_T=%d",
                    holdout_months, holdout_b, md_train.T, md_hold.T)

    # WF folds on training portion
    wf = WfConfig(
        train_months=float(args.train_months),
        test_months=float(args.test_months),
        step_months=float(args.step_months),
        mode=str(args.wf_mode),
    )
    folds = make_folds(md_train.T, wf, bar_ms)
    if len(folds) == 0:
        raise RuntimeError(
            f"Not enough data for walk-forward. T_train={md_train.T} bars; "
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

    # MA cache (full md; causal, safe)
    uniq_windows = sorted(set(list(scfg.fast_values) + list(scfg.slow_values)))
    ma_cache: Dict[int, np.ndarray] = {}
    logger.info("Precomputing rolling means for %d unique windows...", len(uniq_windows))
    for w in uniq_windows:
        ma_cache[int(w)] = rolling_mean_matrix(md.close, int(w))

    # optional funding
    funding_full = None
    if bool(args.funding):
        funding_full = bt.load_funding_for_symbols(paths, exchange_id, md.symbols, md.ts_ms)

    # backtest config
    btcfg = bt.BacktestConfig(
        initial_equity=1.0,
        fee_rate=float(args.fee_bps) / 10_000.0,
        slippage_rate=float(args.slippage_bps) / 10_000.0,
        max_gross_leverage=float(args.max_gross),
        max_abs_position=float(args.max_pos),
        scale_to_gross=True,
    )

    fcfg = FilterConfig(
        min_pos_folds=float(args.min_pos_folds),
        min_sharpe_median=float(args.min_sharpe_median),
        min_equity_end_median=float(args.min_eq_median),
        max_dd_worst=float(args.max_dd_worst),
        max_turnover_median=float(args.max_turnover_median),
        min_holdout_equity_end=float(args.min_hold_eq),
        max_holdout_dd=float(args.max_hold_dd),
    )

    out_dir = ensure_reports_dir(paths)

    rows: List[Dict[str, Any]] = []

    for i, (fast, slow) in enumerate(candidates, 1):
        warmup = max(int(fast), int(slow))
        pos_full = build_ma_positions(
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
            md_slice = slice_marketdata(md, ts0, ts1)
            pos_slice = pos_full[ts0:ts1, :].copy()
            pos_slice[0, :] = 0.0  # START TEST FOLD FLAT
            fund_slice = funding_full[ts0:ts1, :] if funding_full is not None else None

            res = bt.run_backtest(md_slice, pos_slice, btcfg, funding_rate=fund_slice)
            m = res.metrics
            fold_sharpe.append(float(m.get("sharpe", float("nan"))))
            fold_cagr.append(float(m.get("cagr", float("nan"))))
            fold_mdd.append(float(m.get("max_drawdown", float("nan"))))
            fold_end.append(float(m.get("equity_end", float("nan"))))
            fold_turn.append(float(m.get("turnover_mean", float("nan"))))

        sh = summarize(fold_sharpe)
        cg = summarize(fold_cagr)
        dd = summarize(fold_mdd)
        ed = summarize(fold_end)
        tn = summarize(fold_turn)

        pos_folds = float(np.mean(np.array(fold_end, dtype=np.float64) > 1.0)) if fold_end else float("nan")
        dd_worst = dd["min"]
        eq_med = ed["median"]
        eq_min = ed["min"]
        sh_med = sh["median"]
        sh_min = sh["min"]
        tn_med = tn["median"]

        s_train = score_train(
            eq_end_median=eq_med,
            eq_end_min=eq_min,
            sharpe_median=sh_med,
            sharpe_min=sh_min,
            dd_worst=dd_worst,
            pos_folds=pos_folds,
            turnover_median=tn_med,
        )
        pass_train = passes_filters(
            fcfg=fcfg,
            pos_folds=pos_folds,
            sharpe_median=sh_med,
            eq_end_median=eq_med,
            dd_worst=dd_worst,
            turnover_median=tn_med,
        )

        rows.append(
            {
                "fast": int(fast),
                "slow": int(slow),
                "folds": int(len(folds)),
                "test_sharpe_median": sh_med,
                "test_sharpe_min": sh_min,
                "test_cagr_median": cg["median"],
                "test_equity_end_median": eq_med,
                "test_equity_end_min": eq_min,
                "test_mdd_worst": dd_worst,
                "test_pos_folds": pos_folds,
                "turnover_mean_median": tn_med,
                "score_train": s_train,
                "pass_train": bool(pass_train),
                "holdout_equity_end": float("nan"),
                "holdout_sharpe": float("nan"),
                "holdout_max_dd": float("nan"),
                "pass_holdout": False,
                "score_final": float("nan"),
                "pass_all": False,
            }
        )

        if i % max(1, len(candidates) // 10) == 0 or i == len(candidates):
            logger.info("Progress %d/%d", i, len(candidates))

    df = pl.DataFrame(rows).sort(["pass_train", "score_train"], descending=True)

    # Holdout eval for top-N
    if md_hold is not None and int(args.holdout_top) > 0:
        topn = min(int(args.holdout_top), df.height)
        logger.info("Evaluating holdout for top-%d candidates...", topn)

        hold_rows: List[Dict[str, Any]] = []
        for r in df.head(topn).to_dicts():
            fast = int(r["fast"])
            slow = int(r["slow"])
            warmup = max(int(fast), int(slow))

            pos_full = build_ma_positions(
                md.close,
                ma_cache[int(fast)],
                ma_cache[int(slow)],
                leverage=scfg.leverage,
                long_only=scfg.long_only,
                warmup=warmup,
            )
            pos_h = pos_full[holdout_start:, :].copy()
            pos_h[0, :] = 0.0

            fund_h = funding_full[holdout_start:, :] if funding_full is not None else None
            res_h = bt.run_backtest(md_hold, pos_h, btcfg, funding_rate=fund_h)
            mh = res_h.metrics

            hold_eq = float(mh.get("equity_end", float("nan")))
            hold_sh = float(mh.get("sharpe", float("nan")))
            hold_dd = float(mh.get("max_drawdown", float("nan")))

            pass_hold = (
                math.isfinite(hold_eq)
                and math.isfinite(hold_dd)
                and (hold_eq >= fcfg.min_holdout_equity_end)
                and (hold_dd >= -abs(fcfg.max_holdout_dd))
            )

            s_final = float(r["score_train"])
            if math.isfinite(hold_eq):
                s_final += 4.0 * (hold_eq - 1.0)
            if math.isfinite(hold_sh):
                s_final += 0.10 * hold_sh
            if math.isfinite(hold_dd):
                s_final -= 0.30 * abs(hold_dd)

            hold_rows.append(
                {
                    "fast": fast,
                    "slow": slow,
                    "holdout_equity_end": hold_eq,
                    "holdout_sharpe": hold_sh,
                    "holdout_max_dd": hold_dd,
                    "pass_holdout": bool(pass_hold),
                    "score_final": s_final,
                }
            )

        hold_df = pl.DataFrame(hold_rows)
        df = df.join(hold_df, on=["fast", "slow"], how="left", suffix="_new").with_columns(
            [
                pl.col("holdout_equity_end_new").fill_null(pl.col("holdout_equity_end")).alias("holdout_equity_end"),
                pl.col("holdout_sharpe_new").fill_null(pl.col("holdout_sharpe")).alias("holdout_sharpe"),
                pl.col("holdout_max_dd_new").fill_null(pl.col("holdout_max_dd")).alias("holdout_max_dd"),
                pl.col("pass_holdout_new").fill_null(pl.col("pass_holdout")).alias("pass_holdout"),
                pl.col("score_final_new").fill_null(pl.col("score_final")).alias("score_final"),
            ]
        ).drop(
            ["holdout_equity_end_new", "holdout_sharpe_new", "holdout_max_dd_new", "pass_holdout_new", "score_final_new"]
        )

        df = df.with_columns((pl.col("pass_train") & pl.col("pass_holdout")).alias("pass_all")).sort(
            ["pass_all", "score_final", "score_train"], descending=True
        )
    else:
        df = df.with_columns(pl.col("pass_train").alias("pass_all")).sort(["pass_all", "score_train"], descending=True)

    out = df
    if bool(args.require_pass):
        out = df.filter(pl.col("pass_all") == True)

    show_cols = [
        "fast",
        "slow",
        "pass_all",
        "score_train",
        "test_equity_end_median",
        "test_equity_end_min",
        "test_sharpe_median",
        "test_sharpe_min",
        "test_mdd_worst",
        "test_pos_folds",
        "turnover_mean_median",
    ]
    if md_hold is not None:
        show_cols += ["pass_holdout", "holdout_equity_end", "holdout_sharpe", "holdout_max_dd", "score_final"]

    k = min(int(args.top_k), out.height)
    print(out.head(k).select(show_cols))

    out_parquet = out_dir / f"results_{exchange_id}_{timeframe}_{scfg.strategy}.parquet"
    df.write_parquet(out_parquet, compression="zstd")

    meta = {
        "exchange": exchange_id,
        "timeframe": timeframe,
        "symbols": md.symbols,
        "T": md.T,
        "N": md.N,
        "holdout": {
            "enabled": bool(md_hold is not None),
            "months": holdout_months,
            "start_index": int(holdout_start),
            "train_T": int(md_train.T),
            "hold_T": int(md_hold.T) if md_hold is not None else 0,
            "holdout_top": int(args.holdout_top),
        },
        "wf": {
            "mode": wf.mode,
            "train_months": wf.train_months,
            "test_months": wf.test_months,
            "step_months": wf.step_months,
            "folds": len(folds),
        },
        "filters": {
            "min_pos_folds": fcfg.min_pos_folds,
            "min_sharpe_median": fcfg.min_sharpe_median,
            "min_equity_end_median": fcfg.min_equity_end_median,
            "max_dd_worst": fcfg.max_dd_worst,
            "max_turnover_median": fcfg.max_turnover_median,
            "min_holdout_equity_end": fcfg.min_holdout_equity_end,
            "max_holdout_dd": fcfg.max_holdout_dd,
            "require_pass": bool(args.require_pass),
        },
        "outputs": {"parquet": str(out_parquet)},
        "created_at_utc": utc_now().isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote results: %s", out_parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
