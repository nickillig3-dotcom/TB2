from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from engine_config import DataConfig, SplitConfig
from utils_core import ensure_monotonic_datetime_index, set_global_seed


@dataclass(frozen=True)
class DataSet:
    """Core in-memory format: a time-indexed DataFrame with many feature columns.

    Requirements for engine core:
    - index: pandas.DatetimeIndex, strictly increasing, no duplicates
    - columns: at least 'price' plus any number of feature columns
    """
    frame: pd.DataFrame

    def __post_init__(self):
        ensure_monotonic_datetime_index(self.frame)
        if "price" not in self.frame.columns:
            raise ValueError("DataSet.frame must contain a 'price' column")


def make_synthetic_dataset(cfg: DataConfig, seed: int) -> DataSet:
    """Synthetic, *causal* toy data generator for pipeline validation.

    IMPORTANT: This is a test vector, not a claim about real-world predictability.
    We purposely build a weak but learnable relation:
        r_t depends on features_{t-1} (no lookahead).

    Columns:
      - price
      - f000, f001, ... f{n_features-1}
    """
    set_global_seed(seed)

    idx = pd.date_range(cfg.start, periods=cfg.n_rows, freq=cfg.freq)
    n = cfg.n_rows
    k = cfg.n_features

    # Features ~ N(0,1), mild correlation via shared latent
    latent = np.random.normal(size=(n, 1))
    feats = 0.7 * np.random.normal(size=(n, k)) + 0.3 * latent

    # Hidden "true" weights define the predictable component (causal)
    w_true = np.random.normal(size=(k,))
    w_true = w_true / (np.linalg.norm(w_true) + 1e-12)

    # Returns: noise + small alpha * dot(feats_{t-1}, w_true)
    alpha = 0.0008  # small, for toy signal
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


def split_dataset(ds: DataSet, cfg: SplitConfig) -> Dict[str, DataSet]:
    """Simple chronological split with optional 'embargo' rows around boundaries.

    This is the smallest useful step towards later:
      - walk-forward
      - purged/embargo CV
      - leakage guards

    Embargo is applied by *removing* rows near boundaries:
      train: [0 : train_end - embargo)
      valid: [train_end + embargo : valid_end - embargo)
      test:  [valid_end + embargo : end)
    """
    df = ds.frame
    n = len(df)
    train_end = int(n * cfg.train_frac)
    valid_end = int(n * (cfg.train_frac + cfg.valid_frac))

    emb = int(cfg.embargo_rows)
    emb = max(0, emb)

    t0, t1, t2, t3 = 0, train_end, valid_end, n

    train_slice = df.iloc[t0: max(t0, t1 - emb)]
    valid_slice = df.iloc[min(t3, t1 + emb): max(min(t3, t2 - emb), min(t3, t1 + emb))]
    test_slice = df.iloc[min(t3, t2 + emb): t3]

    out: Dict[str, DataSet] = {
        "train": DataSet(train_slice.copy()),
        "valid": DataSet(valid_slice.copy()),
        "test": DataSet(test_slice.copy()),
    }

    for name, part in out.items():
        if len(part.frame) < cfg.min_rows:
            raise ValueError(
                f"Split '{name}' too small ({len(part.frame)} rows). "
                f"Increase data.n_rows or adjust splits/min_rows/embargo."
            )
    return out
