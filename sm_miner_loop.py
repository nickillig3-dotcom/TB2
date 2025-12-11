#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm_miner_loop.py — einfacher Endlos-Looper für sm_miner.py

Idee:
- Ruft sm_miner.py in einer Schleife mit mode=random auf.
- Jede Runde: neue Zufalls-Seed -> andere (fast,slow)-Kombinationen.
- Ergebnisse landen ganz normal unter reports/miner_runs/....

Beenden: STRG+C
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import subprocess
import time
from pathlib import Path
from typing import List
import sys

def build_cmd(args: argparse.Namespace, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        "sm_miner.py",
        "mine",
        "--exchange",
        args.exchange,
        "--timeframe",
        args.timeframe,
        "--top-n",
        str(args.top_n),
        "--years",
        str(args.years),
        "--strategy",
        "ma",
        "--mode",
        "random",
        "--samples",
        str(args.samples),
        "--seed",
        str(seed),
        "--fast-min",
        str(args.fast_min),
        "--fast-max",
        str(args.fast_max),
        "--fast-step",
        str(args.fast_step),
        "--slow-min",
        str(args.slow_min),
        "--slow-max",
        str(args.slow_max),
        "--slow-step",
        str(args.slow_step),
        "--leverage",
        str(args.leverage),
        "--train-months",
        str(args.train_months),
        "--test-months",
        str(args.test_months),
        "--step-months",
        str(args.step_months),
        "--wf-mode",
        args.wf_mode,
        "--holdout-months",
        str(args.holdout_months),
        "--holdout-top",
        str(args.holdout_top),
        "--fee-bps",
        str(args.fee_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--max-gross",
        str(args.max_gross),
        "--max-pos",
        str(args.max_pos),
        "--top-k",
        str(args.top_k),
    ]

    if args.funding:
        cmd.append("--funding")
    if args.require_pass:
        cmd.append("--require-pass")
    if args.quiet:
        cmd.append("--quiet")

    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sm_miner_loop.py",
        description="Endlos-Loop für sm_miner.py (random search in Batches)",
    )

    # Basis (muss zu deinem Datenset passen)
    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--years", type=float, default=3.0)
    p.add_argument("--top-n", type=int, default=2)

    # Search Space für MA
    p.add_argument("--fast-min", type=int, default=12)
    p.add_argument("--fast-max", type=int, default=96)
    p.add_argument("--fast-step", type=int, default=12)
    p.add_argument("--slow-min", type=int, default=60)
    p.add_argument("--slow-max", type=int, default=720)
    p.add_argument("--slow-step", type=int, default=30)

    # Walk-Forward / Holdout
    p.add_argument("--train-months", type=float, default=12.0)
    p.add_argument("--test-months", type=float, default=6.0)
    p.add_argument("--step-months", type=float, default=3.0)
    p.add_argument("--wf-mode", default="rolling", choices=["rolling", "expanding"])
    p.add_argument("--holdout-months", type=float, default=12.0)
    p.add_argument("--holdout-top", type=int, default=50)

    # Kosten / Limits
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--max-pos", type=float, default=1.0)
    p.add_argument("--funding", action="store_true")

    # Filter / Output
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # Loop-Settings
    p.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Anzahl random Kandidaten pro sm_miner-Run",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Wie viele Runs? 0 = unendlich (bis STRG+C).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="Pause in Sekunden zwischen den Runs.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=== sm_miner_loop: starte Endlos-Suche ===")
    print(f"Exchange   : {args.exchange}")
    print(f"Timeframe  : {args.timeframe}")
    print(f"Years      : {args.years}")
    print(f"Top-N      : {args.top_n if hasattr(args, 'top-n') else args.top_n}")
    print(f"Fast-Range : {args.fast_min}..{args.fast_max} step {args.fast_step}")
    print(f"Slow-Range : {args.slow_min}..{args.slow_max} step {args.slow_step}")
    print(f"WF         : train={args.train_months}m test={args.test_months}m step={args.step_months}m")
    print(f"Holdout    : {args.holdout_months}m (top {args.holdout_top})")
    print(f"Samples    : {args.samples} Kandidaten pro Run")
    print(f"Runs       : {args.runs} (0 = unendlich)")
    print(f"Sleep      : {args.sleep} Sekunden")
    print("=========================================")

    run_idx = 0
    while args.runs == 0 or run_idx < args.runs:
        run_idx += 1
        seed = random.randint(1, 2**31 - 1)

        ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[loop] Run {run_idx} | seed={seed} | {ts}", flush=True)

        cmd = build_cmd(args, seed)
        print("[loop] Command:", " ".join(cmd), flush=True)

        try:
            # Wir lassen Fehler durchlaufen, brechen aber nicht die ganze Schleife ab
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\n[loop] KeyboardInterrupt -> Ende.")
            return 0
        except Exception as e:
            print(f"[loop] Fehler in Run {run_idx}: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\n[loop] Fertig, maximale Anzahl Runs erreicht.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
