from __future__ import annotations

from pathlib import Path

import polars as pl


RUNS_DIR = Path("reports") / "miner_runs"
OUT_ALL = Path("reports") / "miner_global.parquet"
OUT_HOF = Path("reports") / "miner_hof.parquet"


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

    # === NaN/Null in Holdout rausfiltern ===
    cols = set(all_df.columns)
    if "holdout_equity_end" in cols:
        all_df = all_df.filter(
            pl.col("holdout_equity_end").is_not_null()
            & ~pl.col("holdout_equity_end").is_nan()
        )

    all_df.write_parquet(OUT_ALL)
    print(f"[merge] Wrote {OUT_ALL} | rows={all_df.height} | cols={len(all_df.columns)}")

    # === Global Top 20 ===
    sort_col = "score_final" if "score_final" in cols else (
        "score_train" if "score_train" in cols else None
    )

    df_top = all_df
    if sort_col is not None:
        df_top = df_top.sort(sort_col, descending=True)

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

    print("[merge] Top 20 Kandidaten (global):")
    print(df_top.select(preview_cols).head(20))

    # === Hall of Fame Filter ===
    if not {"holdout_equity_end", "holdout_sharpe", "holdout_max_dd", "test_pos_folds"}.issubset(
        all_df.columns
    ):
        print("[hof] Nicht alle benötigten Spalten vorhanden, überspringe HoF.")
        return 0

    hof = (
        all_df
        .filter(
            (pl.col("holdout_equity_end") > 1.10)    # > +10 % im Holdout
            & (pl.col("holdout_sharpe") > 0.4)      # vernünftige Sharpe im Holdout
            & (pl.col("holdout_max_dd") > -0.60)    # kein >60 % Drawdown im Holdout
            & (pl.col("test_pos_folds") >= 0.66)    # >= 2/3 der WF-Folds positiv
        )
    )

    if hof.is_empty():
        print("[hof] Keine Kandidaten erfüllen den Hall-of-Fame-Filter.")
        return 0

    hof = hof.sort("score_final", descending=True)
    hof.write_parquet(OUT_HOF)

    print(f"[hof] Wrote {OUT_HOF} | rows={hof.height}")
    print("[hof] Top 10 Hall-of-Fame-Kandidaten:")
    print(
        hof.select(preview_cols)
           .head(10)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
