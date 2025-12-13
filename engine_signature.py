from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from engine_datafingerprint import infer_dataset_identity_from_data_dict
from utils_core import stable_hash

EVALSIG_SCHEMA = "evalsig_v1"


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    return dict(getattr(obj, "__dict__", {}) or {})


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys with None values (to keep hashes stable when optional fields are added)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = _drop_none(v)
        else:
            out[k] = v
    return out


def evaluation_signature_from_cfg(cfg: Any) -> Dict[str, Any]:
    """
    Evaluation signature = only what changes evaluation outcome.

    Excludes:
      - run_name, mode
      - persistence.db_path
      - execution settings
      - research selection params (top_k etc.)

    For synthetic: signature remains as before (seed injected, no dataset_id/version keys).
    For non-synthetic: dataset_id + dataset_version are injected (from provided version or file stat).
    """
    data = _as_dict(getattr(cfg, "data", None))
    splits = _as_dict(getattr(cfg, "splits", None))
    cv = _as_dict(getattr(cfg, "cv", None))
    backtest = _as_dict(getattr(cfg, "backtest", None))
    costs = _as_dict(getattr(cfg, "costs", None))

    # keep signature stable for synthetic
    if data.get("type") == "synthetic":
        if hasattr(cfg, "seed"):
            data["seed"] = int(getattr(cfg, "seed"))
    else:
        # inject dataset identity for real data
        dataset_id, dataset_version, _ = infer_dataset_identity_from_data_dict(
            data, seed=int(getattr(cfg, "seed")) if hasattr(cfg, "seed") else None
        )
        if not data.get("dataset_id"):
            data["dataset_id"] = dataset_id
        if not data.get("dataset_version"):
            data["dataset_version"] = dataset_version

    # drop None keys to avoid hash churn on optional fields
    data = _drop_none(data)
    splits = _drop_none(splits)
    cv = _drop_none(cv)
    backtest = _drop_none(backtest)
    costs = _drop_none(costs)

    return {
        "schema": EVALSIG_SCHEMA,
        "data": data,
        "splits": splits,
        "cv": cv,
        "backtest": backtest,
        "costs": costs,
    }


def evaluation_signature_from_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw.get("data") or {})
    splits = dict(raw.get("splits") or {})
    cv = dict(raw.get("cv") or {})
    backtest = dict(raw.get("backtest") or {})
    costs = dict(raw.get("costs") or {})

    if data.get("type") == "synthetic" and "seed" in raw:
        try:
            data["seed"] = int(raw["seed"])
        except Exception:
            pass
    else:
        dataset_id, dataset_version, _ = infer_dataset_identity_from_data_dict(
            data, seed=int(raw.get("seed")) if isinstance(raw.get("seed"), int) else None
        )
        data.setdefault("dataset_id", dataset_id)
        data.setdefault("dataset_version", dataset_version)

    data = _drop_none(data)
    splits = _drop_none(splits)
    cv = _drop_none(cv)
    backtest = _drop_none(backtest)
    costs = _drop_none(costs)

    return {
        "schema": EVALSIG_SCHEMA,
        "data": data,
        "splits": splits,
        "cv": cv,
        "backtest": backtest,
        "costs": costs,
    }


def evaluation_hash_from_signature(sig: Dict[str, Any]) -> str:
    return stable_hash(sig)


def evaluation_hash_from_cfg(cfg: Any) -> str:
    return stable_hash(evaluation_signature_from_cfg(cfg))
