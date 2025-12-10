#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm_data.py — Strategy Miner Data Layer (Perpetual Futures)

Goals:
- Best-practice data ingestion (no CSV): partitioned Parquet datasets (year/month)
- Exchange-agnostic via CCXT (default: Binance USDT-M Perpetuals: binanceusdm)
- Robust: resumable downloads, retries with exponential backoff, structured logging
- Testable: validation command (duplicates/gaps/schema), deterministic file layout

Install:
  pip install -U ccxt pyarrow polars

Examples:
  # Dry-run: show selected top markets
  python sm_data.py download --dry-run

  # Download last 5 years of 1h OHLCV for top 10 USDT linear perps
  python sm_data.py download --years 5 --timeframe 1h --top-n 10 --what ohlcv

  # Download OHLCV + funding (if exchange supports it)
  python sm_data.py download --what ohlcv,funding

  # Validate datasets (duplicates/gaps)
  python sm_data.py validate --timeframe 1h

Data layout:
  data/
    parquet/{exchange}/{timeframe}/{symbol_sanitized}/year=YYYY/month=MM/*.parquet
    state/{exchange}/{timeframe}/{symbol_sanitized}.json
  logs/sm_data.log
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Third-party (required)
try:
    import ccxt  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: ccxt. Install via: pip install -U ccxt") from e

try:
    import polars as pl  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: polars. Install via: pip install -U polars") from e

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.dataset as ds  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: pyarrow. Install via: pip install -U pyarrow") from e


# -----------------------------
# JSON (fast if orjson exists)
# -----------------------------
try:
    import orjson  # type: ignore

    def _json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8")

    def _json_loads(s: str) -> Any:
        return orjson.loads(s)

except Exception:

    def _json_dumps(obj: Any) -> str:
        return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)

    def _json_loads(s: str) -> Any:
        return json.loads(s)


# -----------------------------
# Time utils
# -----------------------------
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
def ms_to_utc_iso(ms: int) -> str:
    """Milliseconds since epoch -> ISO UTC string with 'Z' suffix (timezone-aware)."""
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def to_ms(ts: dt.datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(ts.timestamp() * 1000)


def subtract_years_safe(ts: dt.datetime, years: int) -> dt.datetime:
    """Subtract calendar years safely (handles Feb 29)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    try:
        return ts.replace(year=ts.year - years)
    except ValueError:
        # e.g. Feb 29 -> Feb 28
        return ts.replace(month=2, day=28, year=ts.year - years)


def timeframe_to_ms(timeframe: str) -> int:
    """
    Supports ccxt-like timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w.
    """
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86_400_000
    if tf.endswith("w"):
        return int(tf[:-1]) * 7 * 86_400_000
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


# -----------------------------
# Logging
# -----------------------------
def setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sm_data.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler (always debug)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logging.getLogger("ccxt").setLevel(logging.WARNING)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Paths:
    base: Path = Path("data")
    logs: Path = Path("logs")

    def parquet_dir(self, exchange_id: str, timeframe: str, symbol_safe: str) -> Path:
        return self.base / "parquet" / exchange_id / timeframe / symbol_safe

    def state_path(self, exchange_id: str, timeframe: str, symbol_safe: str) -> Path:
        return self.base / "state" / exchange_id / timeframe / f"{symbol_safe}.json"


@dataclass(frozen=True)
class DownloadConfig:
    exchange_id: str = "binanceusdm"
    timeframe: str = "1h"
    years: int = 5
    top_n: int = 10
    quote: str = "USDT"
    linear_only: bool = True
    swaps_only: bool = True
    what: Tuple[str, ...] = ("ohlcv",)  # "ohlcv", "funding"
    compression: str = "zstd"  # snappy, zstd, gzip...
    ohlcv_limit: int = 1500  # typical max for many exchanges
    funding_limit: int = 1000
    max_retries: int = 8
    retry_base_sleep_s: float = 0.6
    retry_max_sleep_s: float = 20.0
    jitter_s: float = 0.25
    request_timeout_ms: int = 30_000
    enable_rate_limit: bool = True
    verbose: bool = True
    require_full_history: bool = True
    history_tolerance_bars: int = 2

# -----------------------------
# Small helpers
# -----------------------------
def sanitize_symbol(symbol: str) -> str:
    # Safe for directory names; keep it human-readable.
    return (
        symbol.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )
def has_any_parquet(base_dir: Path) -> bool:
    if not base_dir.exists():
        return False
    try:
        next(base_dir.rglob("*.parquet"))
        return True
    except StopIteration:
        return False

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return _json_loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger("state").exception("Failed reading JSON state: %s", path)
        return None


# -----------------------------
# CCXT exchange wrapper
# -----------------------------
def init_exchange(cfg: DownloadConfig) -> Any:
    logger = logging.getLogger("exchange")
    exchange_cls = getattr(ccxt, cfg.exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unknown CCXT exchange id: {cfg.exchange_id!r}")

    ex = exchange_cls(
        {
            "enableRateLimit": cfg.enable_rate_limit,
            "timeout": cfg.request_timeout_ms,
        }
    )

    # Some exchanges benefit from options; keep minimal and explicit.
    # Users can later extend in config (e.g., by env vars).
    logger.info("Initialized exchange=%s, rateLimit=%s ms", cfg.exchange_id, getattr(ex, "rateLimit", "n/a"))
    return ex


def retry_call(
    fn,
    *,
    max_retries: int,
    base_sleep_s: float,
    max_sleep_s: float,
    jitter_s: float,
    what: str,
) -> Any:
    logger = logging.getLogger("retry")
    attempt = 0
    while True:
        try:
            return fn()
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable) as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("%s failed after %d retries: %r", what, max_retries, e)
                raise
            sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            sleep_s = sleep_s + random.random() * jitter_s
            logger.warning("%s transient error (%d/%d): %r | sleeping %.2fs", what, attempt, max_retries, e, sleep_s)
            time.sleep(sleep_s)
        except ccxt.ExchangeError as e:
            # Often non-retryable, but sometimes rate-limit responses come as ExchangeError.
            msg = str(e).lower()
            is_rate = any(k in msg for k in ["rate limit", "too many requests", "429", "limit"])
            attempt += 1
            if (not is_rate) or attempt > max_retries:
                logger.error("%s exchange error: %r", what, e)
                raise
            sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            sleep_s = sleep_s + random.random() * jitter_s
            logger.warning("%s rate-like error (%d/%d): %r | sleeping %.2fs", what, attempt, max_retries, e, sleep_s)
            time.sleep(sleep_s)


# -----------------------------
# Market selection
# -----------------------------
def select_top_perp_markets(ex: Any, cfg: DownloadConfig) -> List[str]:
    """
    "Best 10" is defined as top-N by 24h quoteVolume (if available), else by baseVolume,
    restricted to swap markets and (optionally) linear USDT quote.
    """
    logger = logging.getLogger("markets")

    markets = retry_call(
        lambda: ex.load_markets(),
        max_retries=cfg.max_retries,
        base_sleep_s=cfg.retry_base_sleep_s,
        max_sleep_s=cfg.retry_max_sleep_s,
        jitter_s=cfg.jitter_s,
        what="load_markets",
    )

    candidates: List[Dict[str, Any]] = []
    for m in markets.values():
        try:
            if m.get("active") is False:
                continue
            if cfg.swaps_only and not m.get("swap", False):
                continue
            if cfg.linear_only and m.get("linear") is not True:
                continue
            if cfg.quote and m.get("quote") != cfg.quote:
                continue
            # Many exchanges use settle/contract specifics; keep it strict but robust.
            candidates.append(m)
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f"No candidates found for exchange={cfg.exchange_id} swaps_only={cfg.swaps_only} "
            f"linear_only={cfg.linear_only} quote={cfg.quote}"
        )

    symbols = [m["symbol"] for m in candidates]
    logger.info("Candidate perp markets: %d", len(symbols))

    # Fetch tickers to rank by volume.
    def _normalize_tickers(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            out: Dict[str, Any] = {}
            for it in raw:
                if isinstance(it, dict) and it.get("symbol"):
                    out[str(it["symbol"])] = it
            return out
        return {}

    def _fetch_tickers() -> Dict[str, Any]:
        if getattr(ex, "has", {}).get("fetchTickers"):
            # Avoid query-length / exchange-side symbol limits when list is large.
            if len(symbols) > 200:
                all_t = _normalize_tickers(ex.fetch_tickers())
                return {s: all_t.get(s) for s in symbols if s in all_t}
            try:
                return _normalize_tickers(ex.fetch_tickers(symbols))
            except TypeError:
                all_t = _normalize_tickers(ex.fetch_tickers())
                return {s: all_t.get(s) for s in symbols if s in all_t}
        # fallback: per-symbol fetch_ticker
        out: Dict[str, Any] = {}
        for s in symbols:
            out[s] = ex.fetch_ticker(s)
        return out

    tickers: Dict[str, Any] = retry_call(
        _fetch_tickers,
        max_retries=cfg.max_retries,
        base_sleep_s=cfg.retry_base_sleep_s,
        max_sleep_s=cfg.retry_max_sleep_s,
        jitter_s=cfg.jitter_s,
        what="fetch_tickers",
    )

    ranked: List[Tuple[str, float]] = []
    for s, t in tickers.items():
        if not t:
            continue
        qv = t.get("quoteVolume")
        bv = t.get("baseVolume")
        vol = None
        if isinstance(qv, (int, float)) and math.isfinite(float(qv)):
            vol = float(qv)
        elif isinstance(bv, (int, float)) and math.isfinite(float(bv)):
            vol = float(bv)
        else:
            # Some exchanges hide unified fields; attempt common raw fields:
            info = t.get("info") or {}
            for key in ("quoteVolume", "quote_volume", "turnover", "volValue", "volumeQuote"):
                v = info.get(key)
                try:
                    if v is not None and math.isfinite(float(v)):
                        vol = float(v)
                        break
                except Exception:
                    pass
        if vol is None:
            continue
        ranked.append((s, vol))

    if not ranked:
        raise RuntimeError("Could not rank markets by volume; ticker fields missing/unexpected.")

    ranked.sort(key=lambda x: x[1], reverse=True)

    # Optional: require history coverage for the requested lookback window.
    if cfg.require_full_history and cfg.years > 0:
        start_ms = to_ms(subtract_years_safe(utc_now(), cfg.years))
        tf_ms = timeframe_to_ms(cfg.timeframe)
        tolerance_ms = tf_ms * int(cfg.history_tolerance_bars)
        logger.info(
            "Filtering by history coverage: years=%d timeframe=%s tolerance_bars=%d",
            cfg.years, cfg.timeframe, cfg.history_tolerance_bars
        )

        selected: List[Tuple[str, float]] = []
        for s, vol in ranked:
            if len(selected) >= cfg.top_n:
                break

            def _probe() -> List[List[float]]:
                return ex.fetch_ohlcv(s, timeframe=cfg.timeframe, since=start_ms, limit=2)

            try:
                candles = retry_call(
                    _probe,
                    max_retries=cfg.max_retries,
                    base_sleep_s=cfg.retry_base_sleep_s,
                    max_sleep_s=cfg.retry_max_sleep_s,
                    jitter_s=cfg.jitter_s,
                    what=f"probe_ohlcv({s})",
                )
            except Exception:
                continue

            if not candles:
                continue
            first_ts = int(candles[0][0])
            if first_ts <= start_ms + tolerance_ms:
                selected.append((s, vol))

        if selected:
            top_ranked = selected
            if len(top_ranked) < cfg.top_n:
                logger.warning(
                    "Only %d symbols have >=%d years history at timeframe %s (need %d).",
                    len(top_ranked), cfg.years, cfg.timeframe, cfg.top_n
                )
        else:
            logger.warning("No symbols passed full-history filter; falling back to volume-only selection.")
            top_ranked = ranked[: cfg.top_n]
    else:
        top_ranked = ranked[: cfg.top_n]
    top = [s for s, _ in top_ranked]

    logger.info("Selected top-%d markets by volume:", cfg.top_n)
    for i, (s, vol) in enumerate(top_ranked, 1):
        logger.info("  %2d) %s | quoteVol≈%.3g", i, s, vol)

    return top


# -----------------------------
# Parquet writing (partitioned)
# -----------------------------
def _pa_table_from_polars(df: pl.DataFrame) -> pa.Table:
    # Polars -> Arrow preserves types well.
    return df.to_arrow()


def write_partitioned_parquet(
    df: pl.DataFrame,
    base_dir: Path,
    *,
    compression: str,
    partition_cols: Sequence[str],
    basename_template: str,
) -> None:
    """
    Append-style write: writes new files into partition folders.
    We rely on resume logic to avoid overlapping rows.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    table = _pa_table_from_polars(df)

    ds.write_dataset(
        data=table,
        base_dir=str(base_dir),
        format="parquet",
        partitioning=list(partition_cols),
        existing_data_behavior="overwrite_or_ignore",
        basename_template=basename_template,
        file_options=ds.ParquetFileFormat().make_write_options(compression=compression),
    )


# -----------------------------
# State (resume)
# -----------------------------
def read_resume_since_ms(state_path: Path) -> Optional[int]:
    st = load_json_if_exists(state_path)
    if not st:
        return None
    v = st.get("last_timestamp_ms")
    try:
        if v is None:
            return None
        v = int(v)
        return v
    except Exception:
        return None


def write_state(
    state_path: Path,
    *,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    last_timestamp_ms: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "exchange_id": exchange_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "last_timestamp_ms": int(last_timestamp_ms),
        "updated_at_utc": utc_now().isoformat(),
    }
    if extra:
        payload.update(extra)
    atomic_write_text(state_path, _json_dumps(payload))


# -----------------------------
# Downloaders
# -----------------------------
def download_ohlcv(
    ex: Any,
    *,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    cfg: DownloadConfig,
    out_dir: Path,
    state_path: Path,
) -> None:
    logger = logging.getLogger("ohlcv")

    tf_ms = timeframe_to_ms(timeframe)
    since = start_ms

    # Resume if state exists
    resume = read_resume_since_ms(state_path)
    if resume is not None and not has_any_parquet(out_dir):
        logger.warning("State exists but OHLCV parquet missing/empty at %s -> ignoring state", out_dir)
        resume = None
    if resume is not None and resume >= start_ms:
        since = resume + tf_ms  # continue after last candle
        logger.info("Resume OHLCV %s from %s", symbol, ms_to_utc_iso(since))

    if since >= end_ms:
        logger.warning("Nothing to do for %s (since>=end).", symbol)
        return

    # Buffer -> write in reasonably sized batches
    rows_buffer: List[List[float]] = []
    part_counter = 0
    total_rows = 0

    def flush_buffer() -> None:
        nonlocal rows_buffer, part_counter, total_rows
        if not rows_buffer:
            return

        df = pl.DataFrame(
            rows_buffer,
            schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
            orient="row",
        )

        # Enforce types & sorting
        df = (
            df.with_columns(pl.col("timestamp_ms").cast(pl.Int64))
            .sort("timestamp_ms")
            .unique(subset=["timestamp_ms"], keep="last")
        )

        # Add partitions
        df = df.with_columns(
            [
                (pl.from_epoch(pl.col("timestamp_ms") / 1000, time_unit="s").dt.year()).alias("year"),
                (pl.from_epoch(pl.col("timestamp_ms") / 1000, time_unit="s").dt.month()).alias("month"),
            ]
        )

        last_ts = int(df["timestamp_ms"].max())
        basename = f"ohlcv-part-{part_counter:05d}-{{i}}-{uuid.uuid4().hex}.parquet"
        write_partitioned_parquet(
            df,
            out_dir,
            compression=cfg.compression,
            partition_cols=("year", "month"),
            basename_template=basename,
        )

        total_rows += df.height
        part_counter += 1
        rows_buffer = []

        write_state(
            state_path,
            exchange_id=cfg.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            last_timestamp_ms=last_ts,
            extra={"dataset": "ohlcv"},
        )

        logger.info("Wrote OHLCV batch: %s rows=%d last=%s", symbol, df.height, last_ts)

    logger.info(
        "Download OHLCV %s timeframe=%s range=[%s .. %s)",
        symbol,
        timeframe,
        ms_to_utc_iso(start_ms),
        ms_to_utc_iso(end_ms),
    )

    while since < end_ms:
        def _fetch() -> List[List[float]]:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=cfg.ohlcv_limit)

        ohlcv: List[List[float]] = retry_call(
            _fetch,
            max_retries=cfg.max_retries,
            base_sleep_s=cfg.retry_base_sleep_s,
            max_sleep_s=cfg.retry_max_sleep_s,
            jitter_s=cfg.jitter_s,
            what=f"fetch_ohlcv({symbol})",
        )

        if not ohlcv:
            logger.warning("No more OHLCV returned for %s at since=%d", symbol, since)
            break

        # Filter to [start,end)
        for row in ohlcv:
            # row: [timestamp, open, high, low, close, volume]
            ts = int(row[0])
            if ts < start_ms or ts >= end_ms:
                continue
            rows_buffer.append([ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])

        # Advance since by last timestamp + tf
        last = int(ohlcv[-1][0])
        next_since = last + tf_ms
        if next_since <= since:
            logger.error("Non-advancing since for %s: since=%d last=%d tf_ms=%d", symbol, since, last, tf_ms)
            break
        since = next_since

        # Flush heuristics: write every ~200k rows or when we pass month boundary naturally (buffer size based)
        if len(rows_buffer) >= 200_000:
            flush_buffer()

    flush_buffer()
    logger.info("DONE OHLCV %s total_rows_written≈%d", symbol, total_rows)


def download_funding(
    ex: Any,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    cfg: DownloadConfig,
    out_dir: Path,
    state_path: Path,
) -> None:
    """
    Funding rate history is exchange-specific; CCXT provides fetch_funding_rate_history on some.
    Stored as partitioned Parquet as well.
    """
    logger = logging.getLogger("funding")

    has = getattr(ex, "has", {}) or {}
    if not has.get("fetchFundingRateHistory"):
        logger.warning("Exchange %s does not support fetchFundingRateHistory via CCXT. Skipping funding for %s.", cfg.exchange_id, symbol)
        return

    since = start_ms
    resume = read_resume_since_ms(state_path)
    if resume is not None and not has_any_parquet(out_dir):
        logger.warning("State exists but FUNDING parquet missing/empty at %s -> ignoring state", out_dir)
        resume = None

    # Funding timestamps are not uniform across exchanges; still safe to resume by +1ms
    if resume is not None and resume >= start_ms:
        since = resume + 1
        logger.info("Resume FUNDING %s from %s", symbol, ms_to_utc_iso(since))

    if since >= end_ms:
        logger.warning("Nothing to do for funding %s (since>=end).", symbol)
        return

    rows_buffer: List[Dict[str, Any]] = []
    part_counter = 0
    total_rows = 0

    def flush_buffer() -> None:
        nonlocal rows_buffer, part_counter, total_rows
        if not rows_buffer:
            return
        # Build strict-schema DF only from timestamp_ms + funding_rate (ignore any other keys)
        ts_list = []
        fr_list = []
        for r in rows_buffer:
            t = r.get("timestamp_ms")
            if t is None:
                continue
            ts_list.append(int(t))

            x = r.get("funding_rate")
            try:
                fr_list.append(float(x) if x is not None else None)
            except Exception:
                fr_list.append(None)

        if not ts_list:
            rows_buffer.clear()
            return

        df = (
            pl.DataFrame(
                {"timestamp_ms": ts_list, "funding_rate": fr_list},
                schema={"timestamp_ms": pl.Int64, "funding_rate": pl.Float64},
            )
            .unique(subset=["timestamp_ms"], keep="last")
            .sort("timestamp_ms")
        )

        df = df.with_columns([
            pl.from_epoch("timestamp_ms", time_unit="ms").dt.year().alias("year"),
            pl.from_epoch("timestamp_ms", time_unit="ms").dt.month().alias("month"),
        ])

        last_ts = int(df["timestamp_ms"].max())
        basename = f"funding-part-{part_counter:05d}-{{i}}-{uuid.uuid4().hex}.parquet"
        write_partitioned_parquet(
            df,
            out_dir,
            compression=cfg.compression,
            partition_cols=("year", "month"),
            basename_template=basename,
        )
        total_rows += df.height
        part_counter += 1
        rows_buffer = []
        write_state(
            state_path,
            exchange_id=cfg.exchange_id,
            symbol=symbol,
            timeframe="funding",
            last_timestamp_ms=last_ts,
            extra={"dataset": "funding"},
        )
        logger.info("Wrote FUNDING batch: %s rows=%d last=%s", symbol, df.height, last_ts)

    logger.info(
        "Download FUNDING %s range=[%s .. %s)",
        symbol,
        ms_to_utc_iso(start_ms),
        ms_to_utc_iso(end_ms),
    )

    while since < end_ms:
        def _fetch() -> List[Dict[str, Any]]:
            # CCXT returns list of funding rate entries; exact keys vary.
            return ex.fetch_funding_rate_history(symbol, since=since, limit=cfg.funding_limit)

        items: List[Dict[str, Any]] = retry_call(
            _fetch,
            max_retries=cfg.max_retries,
            base_sleep_s=cfg.retry_base_sleep_s,
            max_sleep_s=cfg.retry_max_sleep_s,
            jitter_s=cfg.jitter_s,
            what=f"fetch_funding_rate_history({symbol})",
        )

        if not items:
            logger.warning("No more funding returned for %s at since=%d", symbol, since)
            break

        # Normalize: timestamp + fundingRate, markPrice, indexPrice if present
        max_seen = None
        for it in items:
            ts = it.get("timestamp")
            if ts is None:
                # try nested/raw
                info = it.get("info") or {}
                ts = info.get("fundingTime") or info.get("time") or info.get("timestamp")
            if ts is None:
                continue
            ts = int(ts)
            if ts < start_ms or ts >= end_ms:
                continue

            rate = it.get("fundingRate")
            if rate is None:
                info = it.get("info") or {}
                rate = info.get("fundingRate") or info.get("funding_rate")
            try:
                rate_f = float(rate) if rate is not None else None
            except Exception:
                rate_f = None

            row = {
                "timestamp_ms": ts,
                "funding_rate": rate_f,
            }
            for k in ("markPrice", "indexPrice", "interestRate"):
                v = it.get(k)
                if v is None:
                    v = (it.get("info") or {}).get(k)
                try:
                    row[k.lower()] = float(v) if v is not None else None
                except Exception:
                    row[k.lower()] = None

            rows_buffer.append(row)
            if max_seen is None or ts > max_seen:
                max_seen = ts

        # Advance since by max_seen + 1ms to avoid duplicates
        if max_seen is None:
            break
        next_since = int(max_seen) + 1
        if next_since <= since:
            logger.error("Non-advancing since for funding %s: since=%d max_seen=%d", symbol, since, max_seen)
            break
        since = next_since

        if len(rows_buffer) >= 200_000:
            flush_buffer()

    flush_buffer()
    logger.info("DONE FUNDING %s total_rows_written≈%d", symbol, total_rows)


# -----------------------------
# Validation
# -----------------------------
def scan_dataset(dir_path: Path) -> pl.LazyFrame:
    # Hive-style partition discovery works well with polars scan_parquet.
    # It will include partition columns if present in file paths (year=, month=)
    return pl.scan_parquet(str(dir_path / "**" / "*.parquet"))

def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """Polars: `streaming=` deprecated -> use `engine=` when available."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)
    
def validate_symbol_timeframe(paths: Paths, cfg: DownloadConfig, symbol: str) -> int:
    logger = logging.getLogger("validate")
    symbol_safe = sanitize_symbol(symbol)

    out_dir = paths.parquet_dir(cfg.exchange_id, cfg.timeframe, symbol_safe)
    if not out_dir.exists():
        logger.warning("No dataset found: %s", out_dir)
        return 1

    lf = scan_dataset(out_dir).select(
        [
            pl.col("timestamp_ms").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    )

    df = collect_streaming(lf)
    if df.is_empty():
        logger.warning("Empty dataset: %s", out_dir)
        return 1

    df = df.sort("timestamp_ms")
    n = df.height

    # Duplicates
    dup_count = n - df.unique(subset=["timestamp_ms"]).height

    # Gaps (expected uniform candles)
    tf_ms = timeframe_to_ms(cfg.timeframe)
    ts = df["timestamp_ms"].to_list()
    gaps = 0
    max_gap = 0
    for i in range(1, len(ts)):
        d = int(ts[i] - ts[i - 1])
        if d != tf_ms:
            gaps += 1
            if d > max_gap:
                max_gap = d

    logger.info(
        "VALIDATE %s | rows=%d | dups=%d | gap_events=%d | max_gap_ms=%d | path=%s",
        symbol,
        n,
        dup_count,
        gaps,
        max_gap,
        out_dir,
    )
    return 0 if dup_count == 0 else 2


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sm_data.py", description="Strategy Miner - Data Layer (Parquet)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download OHLCV/funding for top-N perpetual markets into Parquet.")
    p_dl.add_argument("--exchange", default="binanceusdm", help="CCXT exchange id (default: binanceusdm)")
    p_dl.add_argument("--timeframe", default="1h", help="OHLCV timeframe (default: 1h)")
    p_dl.add_argument("--years", type=int, default=5, help="Lookback years from now (default: 5)")
    p_dl.add_argument("--top-n", type=int, default=10, help="Number of markets (default: 10)")
    p_dl.add_argument("--quote", default="USDT", help="Quote currency filter (default: USDT)")
    p_dl.add_argument("--linear-only", action="store_true", default=True, help="Only linear swaps (default: true)")
    p_dl.add_argument("--no-linear-only", dest="linear_only", action="store_false", help="Disable linear-only filter")
    p_dl.add_argument("--swaps-only", action="store_true", default=True, help="Only swap markets (default: true)")
    p_dl.add_argument("--no-swaps-only", dest="swaps_only", action="store_false", help="Disable swaps-only filter")
    p_dl.add_argument(
        "--what",
        default="ohlcv",
        help="Comma list: ohlcv,funding (default: ohlcv)",
    )
    p_dl.add_argument("--compression", default="zstd", help="Parquet compression (default: zstd)")
    p_dl.add_argument("--dry-run", action="store_true", help="Only select markets and print plan (no downloads)")
    p_dl.add_argument("--quiet", action="store_true", help="Less console output")
    p_dl.add_argument("--allow-short-history", action="store_true", help="Do not require full history coverage for selection")

    p_val = sub.add_parser("validate", help="Validate stored datasets for duplicates/gaps.")
    p_val.add_argument("--exchange", default="binanceusdm", help="CCXT exchange id (default: binanceusdm)")
    p_val.add_argument("--timeframe", default="1h", help="Timeframe to validate (default: 1h)")
    p_val.add_argument("--symbols", default="", help="Comma list of symbols (default: auto top-n selection)")
    p_val.add_argument("--top-n", type=int, default=10, help="If symbols not provided: select top-n (default: 10)")
    p_val.add_argument("--quote", default="USDT", help="Quote currency filter (default: USDT)")
    p_val.add_argument("--quiet", action="store_true", help="Less console output")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    verbose = not getattr(args, "quiet", False)
    paths = Paths()
    setup_logging(paths.logs, verbose=verbose)

    logger = logging.getLogger("main")

    if args.cmd == "download":
        cfg = DownloadConfig(
            exchange_id=args.exchange,
            timeframe=args.timeframe,
            years=int(args.years),
            top_n=int(args.top_n),
            quote=args.quote,
            linear_only=bool(args.linear_only),
            swaps_only=bool(args.swaps_only),
            what=tuple([w.strip().lower() for w in str(args.what).split(",") if w.strip()]),
            compression=args.compression,
            verbose=verbose,
            require_full_history=not bool(args.allow_short_history),
        )

        ex = init_exchange(cfg)
        top_symbols = select_top_perp_markets(ex, cfg)

        end_ts = utc_now()
        start_ts = subtract_years_safe(end_ts, cfg.years)
        start_ms = to_ms(start_ts)
        end_ms = to_ms(end_ts)

        logger.info("Plan: exchange=%s timeframe=%s years=%d markets=%d what=%s",
                    cfg.exchange_id, cfg.timeframe, cfg.years, len(top_symbols), ",".join(cfg.what))

        if args.dry_run:
            logger.warning("DRY-RUN: no downloads executed.")
            return 0

        for symbol in top_symbols:
            symbol_safe = sanitize_symbol(symbol)

            # OHLCV
            if "ohlcv" in cfg.what:
                out_dir = paths.parquet_dir(cfg.exchange_id, cfg.timeframe, symbol_safe)
                st_path = paths.state_path(cfg.exchange_id, cfg.timeframe, symbol_safe)
                try:
                    download_ohlcv(
                        ex,
                        symbol=symbol,
                        timeframe=cfg.timeframe,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        cfg=cfg,
                        out_dir=out_dir,
                        state_path=st_path,
                    )
                except Exception:
                    logger.exception("OHLCV failed for %s", symbol)

            # Funding
            if "funding" in cfg.what:
                out_dir = paths.parquet_dir(cfg.exchange_id, "funding", symbol_safe)
                st_path = paths.state_path(cfg.exchange_id, "funding", symbol_safe)
                try:
                    download_funding(
                        ex,
                        symbol=symbol,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        cfg=cfg,
                        out_dir=out_dir,
                        state_path=st_path,
                    )
                except Exception:
                    logger.exception("FUNDING failed for %s", symbol)

        logger.info("All downloads finished.")
        return 0

    if args.cmd == "validate":
        cfg = DownloadConfig(
            exchange_id=args.exchange,
            timeframe=args.timeframe,
            top_n=int(args.top_n),
            quote=args.quote,
            verbose=verbose,
            require_full_history=False,
        )

        ex = init_exchange(cfg)

        if args.symbols.strip():
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = select_top_perp_markets(ex, cfg)

        rc = 0
        for s in symbols:
            rc = max(rc, validate_symbol_timeframe(paths, cfg, s))
        return rc

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
