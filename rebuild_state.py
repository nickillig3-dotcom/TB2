from pathlib import Path
import json
import datetime as dt
import polars as pl

EXCHANGE = "binanceusdm"
BASE = Path("data")

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

def sanitize_symbol(symbol: str) -> str:
    return (
        symbol.replace("/", "_")
              .replace(":", "_")
              .replace(" ", "_")
              .replace("-", "_")
              .replace("__", "_")
    )

def max_timestamp_ms(parquet_dir: Path) -> int:
    lf = pl.scan_parquet(str(parquet_dir / "**" / "*.parquet")).select(
        pl.col("timestamp_ms").cast(pl.Int64).max().alias("mx")
    )
    df = collect_streaming(lf)
    mx = df["mx"][0]
    if mx is None:
        raise RuntimeError(f"No timestamp_ms found under {parquet_dir}")
    return int(mx)

def write_state(timeframe: str, symbol: str, dataset: str) -> None:
    sym_safe = sanitize_symbol(symbol)
    parquet_dir = BASE / "parquet" / EXCHANGE / timeframe / sym_safe
    if not parquet_dir.exists():
        print(f"[skip] no parquet dir: {parquet_dir}")
        return

    last_ts = max_timestamp_ms(parquet_dir)
    state_path = BASE / "state" / EXCHANGE / timeframe / f"{sym_safe}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "exchange_id": EXCHANGE,
        "symbol": symbol,
        "timeframe": timeframe,
        "last_timestamp_ms": last_ts,
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": dataset,
    }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ok] wrote {state_path} last_timestamp_ms={last_ts}")

symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

# OHLCV state (timeframe = "1h")
for s in symbols:
    write_state("1h", s, "ohlcv")

# Funding state (timeframe = "funding") – nur wenn funding-parquets existieren
for s in symbols:
    write_state("funding", s, "funding")
