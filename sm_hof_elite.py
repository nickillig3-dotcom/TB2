from __future__ import annotations

from pathlib import Path

import polars as pl


HOF_PATH = Path("reports") / "miner_hof.parquet"
VAL_PATH = Path("reports") / "hof_validate.parquet"
OUT_PATH = Path("reports") / "miner_elite.parquet"


# Thresholds (kannst du später leicht anpassen)
MIN_SCENARIOS = 6          # mindestens so viele (years_bt, symbol_mode)-Kombos
MIN_CAGR_MEAN = 0.08       # >= 8% Durchschnitts-CAGR über alle Szenarien
MIN_SHARPE_MEAN = 0.3      # durchschnittliche Sharpe über alle Szenarien
MIN_DD_WORST = -0.80       # kein Drawdown schlimmer als -80% über alle Szenarien
MIN_CAGR_MIN_BOTH = 0.0    # in keinem years_bt für symbol_mode="both" negativ


def main() -> int:
    if not HOF_PATH.exists():
        raise FileNotFoundError(f"HoF-Datei nicht gefunden: {HOF_PATH}")
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation-Datei nicht gefunden: {VAL_PATH}")

    hof = pl.read_parquet(HOF_PATH)
    val = pl.read_parquet(VAL_PATH)

    if hof.is_empty():
        raise RuntimeError(f"HoF-Datei {HOF_PATH} ist leer.")
    if val.is_empty():
        raise RuntimeError(f"Validation-Datei {VAL_PATH} ist leer.")

    # Basis-Aggregation über alle Szenarien (years_bt × symbol_mode)
    agg_all = (
        val
        .group_by(["fast", "slow"])
        .agg(
            [
                pl.count().alias("scenarios"),
                pl.col("cagr").mean().alias("cagr_mean"),
                pl.col("cagr").min().alias("cagr_min"),
                pl.col("sharpe").mean().alias("sharpe_mean"),
                pl.col("sharpe").min().alias("sharpe_min"),
                # max_drawdown ist negativ; min() = schlimmster Drawdown
                pl.col("max_drawdown").min().alias("dd_worst"),
            ]
        )
    )

    # Spezielle Aggregation nur für symbol_mode="both" (Portfolio BTC+ETH)
    agg_both = (
        val
        .filter(pl.col("symbol_mode") == "both")
        .group_by(["fast", "slow"])
        .agg(
            [
                pl.col("cagr").mean().alias("cagr_mean_both"),
                pl.col("cagr").min().alias("cagr_min_both"),
                pl.col("sharpe").mean().alias("sharpe_mean_both"),
                pl.col("sharpe").min().alias("sharpe_min_both"),
            ]
        )
    )

    # Join mit HoF (um score_final, pass_all etc. zu behalten)
    elite = (
        agg_all
        .join(agg_both, on=["fast", "slow"], how="left")
        .join(hof, on=["fast", "slow"], how="left", suffix="_hof")
    )

    # Filtern nach Elite-Kriterien
    elite_filtered = elite.filter(
        (pl.col("scenarios") >= MIN_SCENARIOS)
        & (pl.col("cagr_mean") >= MIN_CAGR_MEAN)
        & (pl.col("sharpe_mean") >= MIN_SHARPE_MEAN)
        & (pl.col("dd_worst") >= MIN_DD_WORST)
        & (pl.col("cagr_min_both") >= MIN_CAGR_MIN_BOTH)
    )

    if elite_filtered.is_empty():
        print("[elite] Keine Kandidaten erfüllen die Elite-Filter – Thresholds evtl. zu streng.")
        return 0

    # Wenn score_final aus Hof da ist, danach sortieren
    sort_col = "score_final"
    if sort_col not in elite_filtered.columns:
        # falls wegen suffix andere Spalte, fallback
        for c in elite_filtered.columns:
            if c.startswith("score_final"):
                sort_col = c
                break

    elite_sorted = elite_filtered.sort(sort_col, descending=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    elite_sorted.write_parquet(OUT_PATH)

    print(f"[elite] Wrote {OUT_PATH} | rows={elite_sorted.height} | cols={len(elite_sorted.columns)}")
    print("[elite] Top Elite-Kandidaten:")
    preview_cols = [
        "fast",
        "slow",
        "pass_all",
        sort_col,
        "scenarios",
        "cagr_mean",
        "cagr_min",
        "cagr_mean_both",
        "cagr_min_both",
        "sharpe_mean",
        "dd_worst",
    ]
    preview_cols = [c for c in preview_cols if c in elite_sorted.columns]

    print(
        elite_sorted
        .select(preview_cols)
        .head(20)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
