#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm_replay.py — Replay & Report runner for mined strategies.

- Reads a miner results Parquet
- Picks rank-k candidate (optionally only pass_all=True)
- Re-runs backtest on a chosen dataset window
- Writes:
  reports/replay_runs/<timestamp>/metrics.json
  reports/replay_runs/<timestamp>/equity.parquet
  reports/replay_runs/<timestamp>/positions.parquet (optional)

Requires:
  pip install -U numpy polars pyarrow
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
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
    log_path = log_dir / "sm_replay.log"

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
# Paths
# -----------------------------
@dataclass(frozen=True)
class Paths:
    reports: Path = Path("reports")
    logs: Path = Path("logs")

    def replay_out_dir(self) -> Path:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = self.reports / "replay_runs" / ts
        out.mkdir(parents=True, exist_ok=True)
        return out


# -----------------------------
# Helpers
# -----------------------------
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def latest_results_file(reports_dir: Path) -> Path:
    cand = list((reports_dir / "miner_runs").glob("*/*.parquet"))
    if not cand:
        raise FileNotFoundError("No miner results found under reports/miner_runs/*/*.parquet")
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def load_results(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pl.read_parquet(str(path))
    if df.is_empty():
        raise RuntimeError(f"Empty results parquet: {path}")
    return df


def ranking_columns(df: pl.DataFrame) -> List[str]:
    cols = df.columns
    if "score_final" in cols:
        return ["pass_all", "score_final", "score_train"]
    if "score_train" in cols:
        return ["pass_all", "score_train"]
    if "test_sharpe_median" in cols:
        return ["pass_all", "test_sharpe_median"]
    return ["pass_all"] if "pass_all" in cols else []


def select_candidate(df: pl.DataFrame, *, rank: int, only_pass: bool) -> Dict[str, Any]:
    if only_pass and "pass_all" in df.columns:
        df = df.filter(pl.col("pass_all") == True)

    if df.is_empty():
        raise RuntimeError("No candidates available after filtering.")

    # sort
    sort_cols = []
    desc = []
    for c in ranking_columns(df):
        if c in df.columns:
            sort_cols.append(c)
            desc.append(True)

    if sort_cols:
        df = df.sort(sort_cols, descending=desc)

    # rank is 1-based
    idx = int(rank) - 1
    if idx < 0 or idx >= df.height:
        raise IndexError(f"rank={rank} out of range (available={df.height})")

    row = df.row(idx, named=True)
    # required fields for MA
    if "fast" not in row or "slow" not in row:
        raise RuntimeError("Results parquet missing required columns fast/slow.")
    return dict(row)


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


def build_ma_positions(close: np.ndarray, fast: int, slow: int, *, leverage: float, long_only: bool) -> np.ndarray:
    T, N = close.shape
    ma_f = rolling_mean_matrix(close, int(fast))
    ma_s = rolling_mean_matrix(close, int(slow))
    valid = np.isfinite(ma_f) & np.isfinite(ma_s)
    sig_close = np.where(valid & (ma_f > ma_s), 1.0, -1.0)
    if long_only:
        sig_close = np.where(sig_close > 0, 1.0, 0.0)

    pos = np.zeros((T, N), dtype=np.float64)
    pos[1:, :] = sig_close[:-1, :] * float(leverage)

    warm = max(int(fast), int(slow))
    pos[: max(1, warm), :] = 0.0
    return pos


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sm_replay.py", description="Replay & report for mined strategies")
    sub = p.add_subparsers(dest="cmd", required=True)

    plst = sub.add_parser("list", help="List top candidates from a results parquet.")
    plst.add_argument("--results", default="", help="Path to results parquet (default: latest under reports/miner_runs).")
    plst.add_argument("--top", type=int, default=20)
    plst.add_argument("--only-pass", action="store_true")
    plst.add_argument("--quiet", action="store_true")

    run = sub.add_parser("run", help="Replay one candidate and write a report.")
    run.add_argument("--results", default="", help="Path to results parquet (default: latest under reports/miner_runs).")
    run.add_argument("--rank", type=int, default=1, help="1-based rank after sorting (default: 1)")
    run.add_argument("--only-pass", action="store_true", help="Select only pass_all=True candidates.")

    run.add_argument("--exchange", default="binanceusdm")
    run.add_argument("--timeframe", default="1h")
    run.add_argument("--symbols", default="", help='Comma list, e.g. "BTC/USDT:USDT,ETH/USDT:USDT". If empty: discover from state dir.')
    run.add_argument("--top-n", type=int, default=2, help="If symbols empty: use first top-n discovered.")
    run.add_argument("--years", type=float, default=0.0, help="Use last N years from available data (0=all).")
    run.add_argument("--start", default="", help="ISO start (optional)")
    run.add_argument("--end", default="", help="ISO end (optional)")
    run.add_argument("--funding", action="store_true", help="Include funding datasets if present.")

    run.add_argument("--leverage", type=float, default=1.0)
    run.add_argument("--long-only", action="store_true")

    run.add_argument("--fee-bps", type=float, default=4.0)
    run.add_argument("--slippage-bps", type=float, default=0.5)
    run.add_argument("--max-gross", type=float, default=1.0)
    run.add_argument("--max-pos", type=float, default=1.0)

    run.add_argument("--export-positions", action="store_true")
    run.add_argument("--quiet", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    verbose = not bool(args.quiet)

    paths = Paths()
    setup_logging(paths.logs, verbose=verbose)
    logger = logging.getLogger("replay")

    results_path = Path(args.results) if str(getattr(args, "results", "")).strip() else latest_results_file(paths.reports)
    df = load_results(results_path)

    if args.cmd == "list":
        out = df
        if bool(args.only_pass) and "pass_all" in out.columns:
            out = out.filter(pl.col("pass_all") == True)

        # choose display columns if present
        show = []
        for c in [
            "fast",
            "slow",
            "pass_all",
            "score_final",
            "score_train",
            "test_equity_end_median",
            "test_equity_end_min",
            "test_sharpe_median",
            "test_sharpe_min",
            "test_mdd_worst",
            "test_pos_folds",
            "holdout_equity_end",
            "holdout_sharpe",
            "holdout_max_dd",
        ]:
            if c in out.columns:
                show.append(c)

        # sort similarly to miner
        sort_cols = []
        desc = []
        for c in ranking_columns(out):
            if c in out.columns:
                sort_cols.append(c)
                desc.append(True)
        if sort_cols:
            out = out.sort(sort_cols, descending=desc)

        print(out.head(int(args.top)).select(show) if show else out.head(int(args.top)))
        logger.info("Listed candidates from %s", results_path)
        return 0

    if args.cmd == "run":
        cand = select_candidate(df, rank=int(args.rank), only_pass=bool(args.only_pass))
        fast = int(cand["fast"])
        slow = int(cand["slow"])

        exchange_id = args.exchange
        timeframe = args.timeframe
        bpaths = bt.Paths()

        if args.symbols.strip():
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            discovered = bt.discover_symbols_from_state(bpaths, exchange_id, timeframe)
            if not discovered:
                raise RuntimeError(
                    f"No symbols provided and none discovered in {bpaths.state_dir(exchange_id, timeframe)}. "
                    f"Run sm_data.py download first."
                )
            symbols = discovered[: int(args.top_n)]

        # time window
        start_ms = bt.parse_iso_date_to_ms(args.start) if str(args.start).strip() else None
        end_ms = bt.parse_iso_date_to_ms(args.end) if str(args.end).strip() else None

        if args.years and float(args.years) > 0 and (start_ms is None and end_ms is None):
            bounds = [bt.dataset_time_bounds(bpaths, exchange_id, timeframe, s) for s in symbols]
            end_ms_eff = min(mx for _, mx in bounds)
            start_ms_eff = int(end_ms_eff - (float(args.years) * 365.25 * 24 * 3600 * 1000))
            start_ms, end_ms = start_ms_eff, end_ms_eff
            logger.info("Window from available data: start=%d end=%d (years=%.2f)", start_ms, end_ms, float(args.years))

        series_list = [
            bt.load_ohlcv_symbol(bpaths, exchange_id, timeframe, s, start_ms=start_ms, end_ms=end_ms)
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

        pos = build_ma_positions(md.close, fast, slow, leverage=float(args.leverage), long_only=bool(args.long_only))

        cfg = bt.BacktestConfig(
            initial_equity=1.0,
            fee_rate=float(args.fee_bps) / 10_000.0,
            slippage_rate=float(args.slippage_bps) / 10_000.0,
            max_gross_leverage=float(args.max_gross),
            max_abs_position=float(args.max_pos),
            scale_to_gross=True,
        )

        funding = None
        if bool(args.funding):
            funding = bt.load_funding_for_symbols(bpaths, exchange_id, md.symbols, md.ts_ms)

        res = bt.run_backtest(md, pos, cfg, funding_rate=funding)

        out_dir = paths.replay_out_dir()
        metrics = dict(res.metrics)
        metrics.update(
            {
                "exchange": exchange_id,
                "timeframe": timeframe,
                "symbols": md.symbols,
                "bars": md.T,
                "candidate": {"strategy": "ma", "fast": fast, "slow": slow, "leverage": float(args.leverage), "long_only": bool(args.long_only)},
                "costs": {"fee_bps": float(args.fee_bps), "slippage_bps": float(args.slippage_bps)},
                "source_results": str(results_path),
                "created_at_utc": utc_now().isoformat(),
            }
        )

        # Print + write
        print(json.dumps(metrics, indent=2, sort_keys=True))
        write_json(out_dir / "metrics.json", metrics)

        eq_df = pl.DataFrame({"timestamp_ms": res.ts_ms.astype(np.int64), "equity": res.equity.astype(np.float64)})
        eq_df.write_parquet(out_dir / "equity.parquet", compression="zstd")

        if bool(args.export_positions):
            # positions aligned to md.ts_ms (T,N); store with wide columns
            cols = {"timestamp_ms": md.ts_ms.astype(np.int64)}
            for j, sym in enumerate(md.symbols):
                cols[f"pos_{bt.sanitize_symbol(sym)}"] = res.positions[:, j].astype(np.float64)
            pl.DataFrame(cols).write_parquet(out_dir / "positions.parquet", compression="zstd")

        logger.info("Wrote replay report to %s", out_dir)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
