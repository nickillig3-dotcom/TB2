from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine_config import DataConfig, SplitConfig
from utils_core import ensure_monotonic_datetime_index, set_global_seed


@dataclass(frozen=True)
class DataSet:
    """
    Core in-memory format: a time-indexed DataFrame with many feature columns.

    Requirements for engine core:
      - index: pandas.DatetimeIndex, strictly increasing, no duplicates
      - columns: at least 'price' plus any number of feature columns
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        ensure_monotonic_datetime_index(self.frame)
        if "price" not in self.frame.columns:
            raise ValueError("DataSet.frame must contain a 'price' column")


def make_synthetic_dataset(cfg: DataConfig, seed: int) -> DataSet:
    """
    Synthetic, *causal* toy data generator for pipeline validation.

    IMPORTANT: This is a test vector, not a claim about real-world predictability.
    We build a weak but learnable relation: r_t depends on features_{t-1} (no lookahead).

    Columns:
      - price
      - f000, f001, ...
    """
    set_global_seed(seed)

    idx = pd.date_range(cfg.start, periods=cfg.n_rows, freq=cfg.freq)
    n = cfg.n_rows
    k = cfg.n_features

    latent = np.random.normal(size=(n, 1))
    feats = 0.7 * np.random.normal(size=(n, k)) + 0.3 * latent

    w_true = np.random.normal(size=(k,))
    w_true = w_true / (np.linalg.norm(w_true) + 1e-12)

    alpha = 0.0008
    noise = np.random.normal(scale=0.01, size=(n,))

    pred = np.zeros(n)
    pred[1:] = feats[:-1].dot(w_true)
    rets = alpha * pred + noise
    price = 100.0 * np.cumprod(1.0 + rets)

    data = pd.DataFrame(index=idx)
    data["price"] = price
    for j in range(k):
        data[f"f{j:03d}"] = feats[:, j]

    return DataSet(frame=data)


def load_csv_dataset(cfg: DataConfig) -> DataSet:
    """
    Load a dataset from CSV.

    Expected:
      - One timestamp column (timestamp_col, default: 'timestamp')
      - One price column (price_col, default: 'price')
      - Feature columns: either explicitly via feature_cols, or auto-selected from numeric cols.

    Output:
      - index = DatetimeIndex (UTC-naive), strictly increasing
      - columns: 'price' + f000..f{n_features-1}
    """
    if not cfg.path:
        raise ValueError("CSV loader: cfg.path is required")

    path = str(cfg.path)
    ts_col = cfg.timestamp_col or "timestamp"
    price_col = cfg.price_col or "price"
    sep = cfg.sep or ","

    dedup = True if cfg.dedup is None else bool(cfg.dedup)
    dropna = True if cfg.dropna is None else bool(cfg.dropna)
    row_limit = cfg.row_limit

    usecols: Optional[List[str]] = None
    if cfg.feature_cols:
        usecols = [ts_col, price_col] + list(cfg.feature_cols)

        # unique preserving order
        seen = set()
        usecols = [c for c in usecols if (c not in seen and not seen.add(c))]

    df = pd.read_csv(path, sep=sep, usecols=usecols)

    if ts_col not in df.columns:
        raise ValueError(f"CSV loader: timestamp_col='{ts_col}' not found in columns={list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"CSV loader: price_col='{price_col}' not found in columns={list(df.columns)}")

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"CSV loader: failed to parse {bad} timestamps in column '{ts_col}'")

    df = df.drop(columns=[ts_col])
    df.index = ts.dt.tz_localize(None)  # store as tz-naive UTC

    # Rename price -> 'price'
    if price_col != "price":
        df = df.rename(columns={price_col: "price"})

    # sort + dedup
    df = df.sort_index()
    if dedup:
        df = df[~df.index.duplicated(keep="first")]
    else:
        if df.index.has_duplicates:
            raise ValueError("CSV loader: index has duplicates and dedup=False")

    # optional light-mode limit: keep the *last* N rows (most recent)
    if row_limit is not None:
        rl = int(row_limit)
        if rl > 0 and len(df) > rl:
            df = df.iloc[-rl:].copy()

    # select features
    if cfg.feature_cols:
        feat_raw = list(cfg.feature_cols)
        missing = [c for c in feat_raw if c not in df.columns]
        if missing:
            raise ValueError(f"CSV loader: feature_cols missing columns={missing}")
        if len(feat_raw) != int(cfg.n_features):
            raise ValueError(
                f"CSV loader: len(feature_cols)={len(feat_raw)} must match data.n_features={cfg.n_features}"
            )
    else:
        # auto: numeric columns except price
        num_cols = []
        for c in df.columns:
            if c == "price":
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                num_cols.append(c)
        num_cols = sorted(num_cols)

        need = int(cfg.n_features)
        if len(num_cols) < need:
            raise ValueError(
                f"CSV loader: found only {len(num_cols)} numeric feature columns, but need {need}. "
                f"Either add columns or set data.feature_cols explicitly."
            )
        feat_raw = num_cols[:need]

    out = df[["price"] + feat_raw].copy()

    # enforce numeric
    out["price"] = out["price"].astype(float)
    for c in feat_raw:
        out[c] = out[c].astype(float)

    if dropna:
        out = out.dropna(axis=0, how="any")

    # map features to canonical f000.. for current dummy strategy pipeline
    rename_map = {c: f"f{i:03d}" for i, c in enumerate(feat_raw)}
    out = out.rename(columns=rename_map)

    return DataSet(frame=out)


def split_dataset(ds: DataSet, cfg: SplitConfig) -> Dict[str, DataSet]:
    """
    Simple chronological split with optional 'embargo' rows around boundaries.

    Embargo is applied by *removing* rows near boundaries:
      train: [0 : train_end - embargo)
      valid: [train_end + embargo : valid_end - embargo)
      test:  [valid_end + embargo : end)
    """
    df = ds.frame
    n = len(df)

    train_end = int(n * cfg.train_frac)
    valid_end = int(n * (cfg.train_frac + cfg.valid_frac))
    emb = max(0, int(cfg.embargo_rows))

    t0, t1, t2, t3 = 0, train_end, valid_end, n

    train_slice = df.iloc[t0 : max(t0, t1 - emb)]
    valid_slice = df.iloc[min(t3, t1 + emb) : max(min(t3, t2 - emb), min(t3, t1 + emb))]
    test_slice = df.iloc[min(t3, t2 + emb) : t3]

    out: Dict[str, DataSet] = {
        "train": DataSet(train_slice.copy()),
        "valid": DataSet(valid_slice.copy()),
        "test": DataSet(test_slice.copy()),
    }

    for name, part in out.items():
        if len(part.frame) < cfg.min_rows:
            raise ValueError(
                f"Split '{name}' too small ({len(part.frame)} rows). "
                f"Increase data length or adjust splits/min_rows/embargo."
            )
    return out
