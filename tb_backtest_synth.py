from __future__ import annotations

"""
tb_backtest_synth.py

Synthetic one-asset backtest on generated price data for an existing TB run.

This script:
- attaches to an existing run_id
- generates synthetic prices
- runs a simple MA-crossover long/flat strategy
- writes equity.csv, trades.json, metrics.json into the run directory on D:\\TB_DATA\\runs\\<run_id>
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from tb_paths import get_run_dir
from tb_logging import get_tb_logger


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TB: synthetic price backtest for an existing run_id"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        type=str,
        help="Existing run_id to attach the synthetic backtest to.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=1000,
        help="Number of synthetic bars to generate (default: 1000).",
    )
    parser.add_argument(
        "--fast",
        type=int,
        default=10,
        help="Fast moving average length (default: 10).",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=50,
        help="Slow moving average length (default: 50). Must be > fast.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic prices (default: 42).",
    )
    return parser.parse_args(argv)


def generate_synthetic_prices(
    n_bars: int,
    seed: int,
    start_price: float = 100.0,
    sigma: float = 0.001,
) -> List[float]:
    """
    Generate a simple log-normal random walk price series.

    Args:
        n_bars: number of bars to generate (must be > 0).
        seed: random seed (deterministic run for a given seed).
        start_price: initial price level.
        sigma: volatility parameter for the Gaussian step.

    Returns:
        List of prices (length n_bars).
    """
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")

    rng = random.Random(seed)
    prices: List[float] = []
    price = start_price

    for _ in range(n_bars):
        prices.append(price)
        # log-normal style random walk
        step = rng.gauss(0.0, sigma)
        price *= math.exp(step)

    return prices


def run_backtest(
    run_id: str,
    bars: int,
    fast: int,
    slow: int,
    seed: int,
) -> int:
    # --- Basic argument validation ---
    if fast <= 0 or slow <= 0:
        print("[TB][ERROR] fast and slow must be positive integers.", file=sys.stderr)
        return 1
    if slow <= fast:
        print("[TB][ERROR] slow must be greater than fast.", file=sys.stderr)
        return 1
    if bars <= slow + 5:
        print(
            "[TB][ERROR] bars must be greater than slow + 5 "
            f"(got bars={bars}, slow={slow}).",
            file=sys.stderr,
        )
        return 1

    # --- Attach to existing run directory ---
    run_dir = get_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"[TB][ERROR] Run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    log_file = run_dir / "tb.log"
    logger = get_tb_logger("tb.backtest", log_file=log_file)

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        msg = f"run_meta.json not found in run_dir={run_dir}"
        logger.error(msg)
        print(f"[TB][ERROR] {msg}", file=sys.stderr)
        return 1

    logger.info(
        "Starting synthetic backtest: run_id=%s, bars=%d, fast=%d, slow=%d, seed=%d",
        run_id,
        bars,
        fast,
        slow,
        seed,
    )

    # --- Generate synthetic price series ---
    try:
        prices = generate_synthetic_prices(n_bars=bars, seed=seed)
    except Exception as exc:
        logger.error("Failed to generate synthetic prices: %s", exc)
        print(f"[TB][ERROR] Failed to generate synthetic prices: {exc}", file=sys.stderr)
        return 1

    # --- Strategy state ---
    fast_window = deque(maxlen=fast)
    slow_window = deque(maxlen=slow)

    position = 0  # 0=flat, 1=long
    entry_price: float | None = None
    entry_idx: int | None = None

    realized_pnl = 0.0
    equity_series: List[float] = []
    price_series: List[float] = []
    position_series: List[int] = []
    trades: List[Dict[str, Any]] = []

    prev_ma_diff: float | None = None

    # --- Main backtest loop ---
    for idx, price in enumerate(prices):
        fast_window.append(price)
        slow_window.append(price)

        ma_fast = None
        ma_slow = None
        ma_diff = None

        # Only compute signals once the slow MA is "fully" defined
        if len(slow_window) == slow:
            ma_fast = sum(fast_window) / len(fast_window)
            ma_slow = sum(slow_window) / len(slow_window)
            ma_diff = ma_fast - ma_slow

        signal_long = False
        signal_flat = False

        # MA crossover logic: long when fast crosses above slow; flat when fast crosses below slow.
        if ma_diff is not None and prev_ma_diff is not None:
            if prev_ma_diff <= 0.0 and ma_diff > 0.0:
                signal_long = True
            elif prev_ma_diff >= 0.0 and ma_diff < 0.0:
                signal_flat = True

        # Force exit at the last bar if still in position (mark-to-market close).
        if idx == bars - 1 and position == 1:
            signal_flat = True

        # 1) Process exits
        if (
            signal_flat
            and position == 1
            and entry_price is not None
            and entry_idx is not None
        ):
            exit_price = price
            exit_idx = idx
            pnl = exit_price - entry_price
            realized_pnl += pnl

            trades.append(
                {
                    "direction": "long",
                    "qty": 1.0,
                    "entry_idx": entry_idx,
                    "exit_idx": exit_idx,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "holding_period_bars": exit_idx - entry_idx,
                }
            )

            position = 0
            entry_price = None
            entry_idx = None

        # 2) Process entries
        if signal_long and position == 0:
            position = 1
            entry_price = price
            entry_idx = idx

        # 3) Mark-to-market equity
        if position == 1 and entry_price is not None:
            unrealized = price - entry_price
        else:
            unrealized = 0.0

        equity = realized_pnl + unrealized

        equity_series.append(equity)
        price_series.append(price)
        position_series.append(position)

        if ma_diff is not None:
            prev_ma_diff = ma_diff

    # --- Metrics: returns, max drawdown, sharpe-like ---
    returns: List[float] = []
    for i in range(1, len(equity_series)):
        returns.append(equity_series[i] - equity_series[i - 1])

    if equity_series:
        peak = equity_series[0]
        max_drawdown = 0.0
        for eq in equity_series:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_drawdown:
                max_drawdown = dd
        final_equity = equity_series[-1]
    else:
        max_drawdown = 0.0
        final_equity = 0.0

    n_trades = len(trades)
    gross_pnl = realized_pnl
    avg_pnl_per_trade = gross_pnl / n_trades if n_trades > 0 else 0.0
    win_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(win_trades) / n_trades if n_trades > 0 else 0.0

    sharpe_like = 0.0
    if len(returns) >= 2:
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        if std_r > 0:
            sharpe_like = (mean_r / std_r) * math.sqrt(len(returns))

    logger.info(
        "Backtest completed: n_trades=%d, gross_pnl=%.6f, final_equity=%.6f, "
        "max_drawdown=%.6f, win_rate=%.2f, sharpe_like=%.3f",
        n_trades,
        gross_pnl,
        final_equity,
        max_drawdown,
        win_rate,
        sharpe_like,
    )

    logger.info(
        "Artefacts: equity.csv, trades.json, metrics.json written in run_dir=%s",
        run_dir,
    )

    # --- Persist artefacts into the run directory ---
    equity_path = run_dir / "equity.csv"
    trades_path = run_dir / "trades.json"
    metrics_path = run_dir / "metrics.json"

    # equity.csv
    try:
        with equity_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time_idx", "price", "position", "equity"])
            for idx, (price, pos, eq) in enumerate(
                zip(price_series, position_series, equity_series)
            ):
                writer.writerow([idx, idx, f"{price:.8f}", pos, f"{eq:.8f}"])
    except Exception as exc:
        logger.error("Failed to write equity.csv: %s", exc)
        print(f"[TB][ERROR] Failed to write equity.csv: {exc}", file=sys.stderr)
        return 1

    # trades.json
    try:
        with trades_path.open("w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as exc:
        logger.error("Failed to write trades.json: %s", exc)
        print(f"[TB][ERROR] Failed to write trades.json: {exc}", file=sys.stderr)
        return 1

    # metrics.json
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "params": {
            "bars": bars,
            "fast": fast,
            "slow": slow,
            "seed": seed,
            "start_price": 100.0,
            "sigma": 0.001,
        },
        "pnl": {
            "gross_pnl": gross_pnl,
            "n_trades": n_trades,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "win_rate": win_rate,
        },
        "equity": {
            "final_equity": final_equity,
            "max_drawdown": max_drawdown,
            "sharpe_like": sharpe_like,
        },
    }

    try:
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception as exc:
        logger.error("Failed to write metrics.json: %s", exc)
        print(f"[TB][ERROR] Failed to write metrics.json: {exc}", file=sys.stderr)
        return 1

    return 0


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    return run_backtest(
        run_id=args.run_id,
        bars=args.bars,
        fast=args.fast,
        slow=args.slow,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
