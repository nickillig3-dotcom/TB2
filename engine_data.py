from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd

from engine_config import EngineConfig, DataConfig


logger = logging.getLogger(__name__)


class DataSource(ABC):
    """
    Base interface for a data source.

    A data source returns a DataFrame with a DatetimeIndex.
    Columns should be raw (unprefixed); DataLoader will apply a prefix.
    """
    name: str
    prefix: str

    @abstractmethod
    def load(self, engine_cfg: EngineConfig, data_cfg: DataConfig) -> pd.DataFrame:
        raise NotImplementedError


class SyntheticPriceSource(DataSource):
    """
    Generates a synthetic OHLCV-like price stream (actually close + volume).

    This makes the engine runnable out-of-the-box without external files.
    """
    name = "synthetic_price"
    prefix = "price"

    def load(self, engine_cfg: EngineConfig, data_cfg: DataConfig) -> pd.DataFrame:
        n = data_cfg.max_rows
        if n is None:
            n = 50_000 if engine_cfg.mode == "full" else 3_000

        rng = np.random.default_rng(engine_cfg.random_seed)

        idx = pd.date_range(
            start=data_cfg.synthetic_start,
            periods=int(n),
            freq=data_cfg.synthetic_freq,
            tz=data_cfg.tz,
        )

        # Regime-like volatility: switch every ~250 bars
        regimes = rng.integers(low=0, high=3, size=len(idx))
        vol = np.select(
            [regimes == 0, regimes == 1, regimes == 2],
            [0.006, 0.012, 0.02],
            default=0.01,
        )
        drift = np.select(
            [regimes == 0, regimes == 1, regimes == 2],
            [0.0002, 0.0, -0.0001],
            default=0.0,
        )
        rets = rng.normal(loc=drift, scale=vol, size=len(idx))

        close = 100.0 * np.exp(np.cumsum(rets))
        volume = rng.lognormal(mean=10.0, sigma=0.35, size=len(idx))

        df = pd.DataFrame({"close": close, "volume": volume}, index=idx)
        df.index.name = "timestamp"
        return df


class CSVPriceSource(DataSource):
    """
    Loads a price series from a CSV.

    Expected columns by default:
      - timestamp (parseable datetime)
      - close (float)

    Configurable via DataConfig.timestamp_col/close_col.
    """
    name = "csv_price"
    prefix = "price"

    def load(self, engine_cfg: EngineConfig, data_cfg: DataConfig) -> pd.DataFrame:
        path = data_cfg.price_csv_path
        if not path:
            raise ValueError("DataConfig.price_csv_path must be set for CSVPriceSource")

        df = pd.read_csv(path)
        if data_cfg.timestamp_col not in df.columns:
            raise ValueError(f"CSV missing timestamp_col='{data_cfg.timestamp_col}'")
        if data_cfg.close_col not in df.columns:
            raise ValueError(f"CSV missing close_col='{data_cfg.close_col}'")

        df[data_cfg.timestamp_col] = pd.to_datetime(df[data_cfg.timestamp_col], utc=True, errors="coerce")
        df = df.dropna(subset=[data_cfg.timestamp_col])
        df = df.set_index(data_cfg.timestamp_col)
        if data_cfg.tz:
            # Convert to desired tz
            df.index = df.index.tz_convert(data_cfg.tz)

        cols = [data_cfg.close_col]
        if data_cfg.volume_col and data_cfg.volume_col in df.columns:
            cols.append(data_cfg.volume_col)

        df = df[cols].copy()
        df = df.sort_index()

        # Apply resource cap
        if data_cfg.max_rows is not None and len(df) > data_cfg.max_rows:
            df = df.tail(data_cfg.max_rows) if data_cfg.tail else df.head(data_cfg.max_rows)

        # Normalize column names
        rename = {data_cfg.close_col: "close"}
        if data_cfg.volume_col:
            rename[data_cfg.volume_col] = "volume"
        df = df.rename(columns=rename)

        df.index.name = "timestamp"
        return df


@dataclass
class DataBundle:
    """
    Holds the merged dataset and lightweight metadata.
    """
    df: pd.DataFrame
    meta: Dict[str, object]


class DataLoader:
    """
    Loads and merges multiple DataSource objects into a single DataFrame.

    - Columns are prefixed per source (e.g. price_close, ob_imbalance, news_sentiment).
    - Resource caps (max_rows) are handled at the source level where possible.
    """
    def __init__(self, sources: Sequence[DataSource]):
        self.sources = list(sources)

    @staticmethod
    def _apply_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [f"{prefix}_{c}" for c in df.columns]
        return df

    def load(self, engine_cfg: EngineConfig, data_cfg: DataConfig) -> DataBundle:
        parts: List[pd.DataFrame] = []
        meta: Dict[str, object] = {}

        for src in self.sources:
            logger.info("Loading source: %s", getattr(src, "name", src.__class__.__name__))
            dfi = src.load(engine_cfg, data_cfg)
            dfi = self._apply_prefix(dfi, getattr(src, "prefix", src.__class__.__name__))
            parts.append(dfi)
            meta[f"source_{getattr(src, 'prefix', src.__class__.__name__)}_rows"] = len(dfi)

        if not parts:
            raise ValueError("No sources configured")

        # Outer join to preserve all timestamps; later we can support asof-joins/resampling per source.
        df = parts[0]
        for p in parts[1:]:
            df = df.join(p, how="outer")

        df = df.sort_index()

        if data_cfg.ffill:
            df = df.ffill()

        # Safety cap at the end (in case merging expanded rows)
        if data_cfg.max_rows is not None and len(df) > data_cfg.max_rows:
            df = df.tail(data_cfg.max_rows) if data_cfg.tail else df.head(data_cfg.max_rows)

        meta["rows"] = len(df)
        meta["cols"] = len(df.columns)
        meta["start"] = str(df.index.min())
        meta["end"] = str(df.index.max())
        return DataBundle(df=df, meta=meta)


def default_data_loader(data_cfg: DataConfig) -> DataLoader:
    """
    Convenience factory: returns a DataLoader with a price source selected by DataConfig.
    """
    if data_cfg.price_source == "synthetic":
        sources: List[DataSource] = [SyntheticPriceSource()]
    elif data_cfg.price_source == "csv":
        sources = [CSVPriceSource()]
    else:
        raise ValueError(f"Unsupported price_source: {data_cfg.price_source}")

    return DataLoader(sources=sources)
