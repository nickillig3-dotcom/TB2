from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def now_utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def set_global_seed(seed: int) -> None:
    """Best-effort global determinism across stdlib + numpy."""
    random.seed(seed)
    np.random.seed(seed)


def _to_jsonable(obj: Any) -> Any:
    """Convert common python/numpy/pandas objects into JSON-serializable structures."""
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()

    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    return obj


def canonical_json(obj: Any) -> str:
    """Stable JSON (sorted keys, no whitespace) for hashing/persistence."""
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(canonical_json(obj).encode("utf-8"))
    return h.hexdigest()


def compute_code_hash(folder: str = ".", algo: str = "sha256") -> str:
    """Hash all *.py files in a folder (single-folder repo) as a cheap code version id."""
    h = hashlib.new(algo)
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".py"):
            continue
        if fn.startswith(". "):
            continue
        path = os.path.join(folder, fn)
        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError:
            continue
        h.update(fn.encode("utf-8"))
        h.update(b"\0")
        h.update(content)
        h.update(b"\0")
    return h.hexdigest()


def env_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def infer_period_seconds(index: pd.Index) -> float:
    """Infer median period length in seconds from a DatetimeIndex."""
    if not isinstance(index, (pd.DatetimeIndex,)):
        raise TypeError("infer_period_seconds expects a pandas.DatetimeIndex")

    if len(index) < 3:
        return 24 * 60 * 60.0

    diffs = index.to_series().diff().dropna().dt.total_seconds()
    med = float(diffs.median()) if len(diffs) else 24 * 60 * 60.0
    if not np.isfinite(med) or med <= 0:
        return 24 * 60 * 60.0
    return med


def infer_periods_per_year(index: pd.Index) -> float:
    period_sec = infer_period_seconds(index)
    return float(SECONDS_PER_YEAR / period_sec)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def ensure_monotonic_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonic increasing")
    if df.index.has_duplicates:
        raise ValueError("DataFrame index must not contain duplicates")
