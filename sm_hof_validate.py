from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import polars as pl


HOF_PATH = Path("reports") / "miner_hof.parquet"
OUT_PATH = Path("reports") / "hof_validate.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sm_hof_validate.py",
        description="Zweitstufiger Robustheitstest für Hall-of-Fame-Strategien.",
    )

    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--years", type=float, nargs="+", default=[3.0, 2.0, 1.0],
                   help="Liste von Zeitfenstern (Jahre) für Re-Backtests, z.B. 3 2 1")
    p.add_argument("--symbol-modes", nargs="+",
                   choices=["both", "btc", "eth"],
                   default=["both"],
                   help="Welche Symbole testen: both (=Top-N), btc, eth")
    p.add_argument("--top-n", type=int, default=2,
                   help="Top-N Märkte, wenn symbol-mode=both verwendet wird.")
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--funding", action="store_true", default=True)

    p.add_argument("--max-candidates", type=int, default=10,
                   help="Maximal so viele HoF-Kandidaten testen (0 = alle).")

    return p.parse_args()


def load_hof(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"HoF-Datei nicht gefunden: {path} – erst sm_miner_merge.py laufen lassen.")
    df = pl.read_parquet(path)
    if df.is_empty():
        raise RuntimeError(f"HoF-Datei {path} enthält keine Zeilen.")
    return df


def parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """
    Nimmt den stdout von sm_backtest.py und extrahiert das letzte JSON-Objekt.
    """
    idx = stdout.rfind("{")
    if idx == -1:
        raise ValueError("Kein JSON im Backtest-Output gefunden.")
    txt = stdout[idx:]
    return json.loads(txt)


def run_backtest(
    exchange: str,
    timeframe: str,
    years: float,
    fast: int,
    slow: int,
    symbol_mode: str,
    top_n: int,
    fee_bps: float,
    slippage_bps: float,
    leverage: float,
    funding: bool,
) -> Dict[str, Any]:
    cmd: List[str] = [
        sys.executable,
        "sm_backtest.py",
        "run",
        "--exchange",
        exchange,
        "--timeframe",
        timeframe,
        "--years",
        str(years),
        "--fast",
        str(fast),
        "--slow",
        str(slow),
        "--fee-bps",
        str(fee_bps),
        "--slippage-bps",
        str(slippage_bps),
        "--leverage",
        str(leverage),
    ]

    if funding:
        cmd.append("--funding")

    if symbol_mode == "both":
        cmd.extend(["--top-n", str(top_n)])
    elif symbol_mode == "btc":
        cmd.extend(["--symbols", "BTC/USDT:USDT"])
    elif symbol_mode == "eth":
        cmd.extend(["--symbols", "ETH/USDT:USDT"])
    else:
        raise ValueError(f"Unbekannter symbol_mode: {symbol_mode}")

    print("[hof-validate] Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if proc.returncode != 0:
        print("[hof-validate] WARN: Backtest returncode", proc.returncode)
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Backtest fehlgeschlagen (returncode={proc.returncode})")

    stats = parse_json_from_stdout(proc.stdout)
    return stats


def main() -> int:
    args = parse_args()

    hof = load_hof(HOF_PATH)

    # Begrenzung der Anzahl Kandidaten
    if args.max_candidates > 0:
        hof = hof.head(args.max_candidates)

    records: List[Dict[str, Any]] = []

    # Sicherstellen, dass die benötigten Spalten existieren
    required_cols = {"fast", "slow"}
    if not required_cols.issubset(set(hof.columns)):
        raise RuntimeError(f"HoF-Datei fehlt Spalten: {required_cols - set(hof.columns)}")

    for idx, row in enumerate(hof.to_dicts()):
        fast = int(row["fast"])
        slow = int(row["slow"])
        pass_all = bool(row.get("pass_all", False))
        score_final = float(row.get("score_final", float("nan")))

        print(f"\n[hof-validate] Kandidat {idx+1}: fast={fast} slow={slow} pass_all={pass_all} score_final={score_final}")

        for years in args.years:
            for symbol_mode in args.symbol_modes:
                try:
                    stats = run_backtest(
                        exchange=args.exchange,
                        timeframe=args.timeframe,
                        years=years,
                        fast=fast,
                        slow=slow,
                        symbol_mode=symbol_mode,
                        top_n=args.top_n,
                        fee_bps=args.fee_bps,
                        slippage_bps=args.slippage_bps,
                        leverage=args.leverage,
                        funding=args.funding,
                    )
                except Exception as e:
                    print(f"[hof-validate] FEHLER für fast={fast} slow={slow} years={years} mode={symbol_mode}: {e}")
                    continue

                rec: Dict[str, Any] = {
                    "fast": fast,
                    "slow": slow,
                    "pass_all": pass_all,
                    "score_final_hof": score_final,
                    "years_bt": years,
                    "symbol_mode": symbol_mode,
                }

                # Wichtige Kennzahlen aus dem Backtest-JSON ziehen
                for k in [
                    "cagr",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "calmar",
                    "equity_end",
                    "equity_start",
                    "vol_annual",
                    "best_month",
                    "worst_month",
                    "pct_positive_months",
                ]:
                    if k in stats:
                        rec[k] = stats[k]

                records.append(rec)

    if not records:
        print("[hof-validate] Keine erfolgreichen Backtests, nichts zu schreiben.")
        return 1

    df_out = pl.DataFrame(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(OUT_PATH)

    print(f"\n[hof-validate] Wrote {OUT_PATH} | rows={df_out.height} | cols={len(df_out.columns)}")
    print("[hof-validate] Beispiel-Preview:")
    print(
        df_out.sort(["fast", "slow", "years_bt", "symbol_mode"])
             .select([
                 "fast",
                 "slow",
                 "years_bt",
                 "symbol_mode",
                 "cagr",
                 "sharpe",
                 "max_drawdown",
             ])
             .head(30)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
