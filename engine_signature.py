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


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            vv = _drop_none(v)
            if vv is None:
                continue
            out[str(k)] = vv
        return out
    if isinstance(obj, list):
        out_list: list[Any] = []
        for x in obj:
            xx = _drop_none(x)
            if xx is None:
                continue
            out_list.append(xx)
        return out_list
    return obj


def _sanitize_data_for_signature(data: Dict[str, Any], *, seed: int | None) -> Dict[str, Any]:
    typ = str(data.get("type") or "unknown")
    d = dict(data)

    if typ == "synthetic":
        if seed is not None:
            d["seed"] = int(seed)
        # keep synthetic params
        return d

    # Non-synthetic: inject dataset identity (portable) + remove path from signature
    dataset_id, dataset_version, _method = infer_dataset_identity_from_data_dict(d, seed=None)
    d["dataset_id"] = dataset_id
    d["dataset_version"] = dataset_version

    # Path should NOT influence evaluation hash
    d.pop("path", None)

    # Remove synthetic-only keys that might exist as defaults in DataConfig
    d.pop("n_rows", None)
    d.pop("freq", None)
    d.pop("start", None)

    return d


def evaluation_signature_from_cfg(cfg: Any) -> Dict[str, Any]:
    """
    Evaluation signature = only what changes the actual evaluation outcome
    for a given StrategySpec on a given dataset/split/cost model.

    Intentionally excludes:
      - run_name
      - mode (light/full)
      - persistence.db_path
      - execution settings (threads/processes)
      - research settings like top_k (selection logic, not evaluation)

    NOTE:
      - For synthetic data, cfg.seed changes the dataset -> included.
      - For file data, dataset_id/version are injected and path is excluded.
    """
    data = _as_dict(getattr(cfg, "data", None))
    splits = _as_dict(getattr(cfg, "splits", None))
    cv = _as_dict(getattr(cfg, "cv", None))
    backtest = _as_dict(getattr(cfg, "backtest", None))
    costs = _as_dict(getattr(cfg, "costs", None))

    seed = None
    if hasattr(cfg, "seed"):
        try:
            seed = int(getattr(cfg, "seed"))
        except Exception:
            seed = None

    data = _sanitize_data_for_signature(data, seed=seed)

    sig: Dict[str, Any] = {
        "schema": EVALSIG_SCHEMA,
        "data": data,
        "splits": splits,
        "cv": cv,
        "backtest": backtest,
        "costs": costs,
    }
    return _drop_none(sig)


def evaluation_signature_from_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Same signature, but from persisted experiments.config_json (dict)."""
    data = dict(raw.get("data") or {})
    splits = dict(raw.get("splits") or {})
    cv = dict(raw.get("cv") or {})
    backtest = dict(raw.get("backtest") or {})
    costs = dict(raw.get("costs") or {})

    seed = None
    if "seed" in raw:
        try:
            seed = int(raw["seed"])
        except Exception:
            seed = None

    data = _sanitize_data_for_signature(data, seed=seed)

    sig: Dict[str, Any] = {
        "schema": EVALSIG_SCHEMA,
        "data": data,
        "splits": splits,
        "cv": cv,
        "backtest": backtest,
        "costs": costs,
    }
    return _drop_none(sig)


def evaluation_hash_from_signature(sig: Dict[str, Any]) -> str:
    return stable_hash(sig)


def evaluation_hash_from_cfg(cfg: Any) -> str:
    return stable_hash(evaluation_signature_from_cfg(cfg))
