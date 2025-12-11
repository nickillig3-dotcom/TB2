from __future__ import annotations

from pathlib import Path
import sys

import polars as pl


RUNS_DIR = Path("reports") / "miner_runs"
OUT_PATH = Path("reports") / "miner_global.parquet"


def main() -> int:
    if not RUNS_DIR.exists():
        print(f"[merge] Kein Ordner gefunden: {RUNS_DIR}")
        return 1

    files = sorted(RUNS_DIR.rglob("*.parquet"))
    if not files:
        print(f"[merge] Keine results_*.parquet unter {RUNS_DIR} gefunden.")
        return 1

    print(f"[merge] Finde {len(files)} Results-Dateien...")
    dfs: list[pl.DataFrame] = []

    for p in files:
        try:
            df = pl.read_parquet(p)
        except Exception as e:
            print(f"[merge] SKIP {p} wegen Fehler beim Lesen: {e}")
            continue

        df = df.with_columns(
            [
                pl.lit(str(p)).alias("source_file"),
            ]
        )
        dfs.append(df)

    if not dfs:
        print("[merge] Keine Datei konnte gelesen werden.")
        return 1

    all_df = pl.concat(dfs, how="diagonal_relaxed")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_df.write_parquet(OUT_PATH)

    print(f"[merge] Wrote {OUT_PATH} | rows={all_df.height} | cols={len(all_df.columns)}")

    # Kurzer Top-Preview
    cols = set(all_df.columns)

    df_top = all_df

    if "holdout_equity_end" in cols:
        df_top = df_top.filter(pl.col("holdout_equity_end").is_not_null())

    sort_col = None
    if "score_final" in cols:
        sort_col = "score_final"
    elif "score_train" in cols:
        sort_col = "score_train"

    if sort_col is not None:
        df_top = df_top.sort(sort_col, descending=True)

    # sinnvolle Spalten für Preview
    preview_cols = [
        "fast",
        "slow",
        "pass_all",
        "score_final",
        "score_train",
        "holdout_equity_end",
        "holdout_sharpe",
        "holdout_max_dd",
        "test_pos_folds",
        "source_file",
    ]
    preview_cols = [c for c in preview_cols if c in df_top.columns]

    df_top = df_top.select(preview_cols).head(20)
    print("[merge] Top 20 Kandidaten (global):")
    print(df_top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
