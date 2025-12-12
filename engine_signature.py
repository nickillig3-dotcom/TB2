from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from utils_core import stable_hash


EVALSIG_SCHEMA = "evalsig_v1"


def evaluation_signature_from_cfg(cfg: Any) -> Dict[str, Any]:
    """
    Evaluation signature = only what changes the actual evaluation outcome
    for a given StrategySpec on a given dataset/split/cost model.

    Intentionally excludes:
      - run_name
      - mode (light/full)
      - persistence.db_path
      - execution.n_jobs / prefer_processes
      - research settings like top_k (selection logic, not evaluation)

    NOTE: For synthetic data, seed changes the generated dataset -> include cfg.seed.
    """
    data = asdict(cfg.data) if is_dataclass(getattr(cfg, "data", None)) else dict(getattr(cfg, "data", {}) or {})
    splits = asdict(cfg.splits) if is_dataclass(getattr(cfg, "splits", None)) else dict(getattr(cfg, "splits", {}) or {})
    cv = asdict(cfg.cv) if hasattr(cfg, "cv") and is_dataclass(cfg.cv) else dict(getattr(cfg, "cv", {}) or {})
    backtest = asdict(cfg.backtest) if is_dataclass(getattr(cfg, "backtest", None)) else dict(getattr(cfg, "backtest", {}) or {})
    costs = asdict(cfg.costs) if is_dataclass(getattr(cfg, "costs", None)) else dict(getattr(cfg, "costs", {}) or {})

    sig: Dict[str, Any] = {
        "schema": EVALSIG_SCHEMA,
        "data": data,
        "splits": splits,
        "cv": cv,
        "backtest": backtest,
        "costs": costs,
    }

    if (data.get("type") == "synthetic") and hasattr(cfg, "seed"):
        sig["data"]["seed"] = int(getattr(cfg, "seed"))

    return sig


def evaluation_signature_from_config_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same signature, but from persisted experiments.config_json (dict).
    Used for backfilling evaluation_hash in existing DBs.
    """
    data = dict(raw.get("data") or {})
    splits = dict(raw.get("splits") or {})
    cv = dict(raw.get("cv") or {})
    backtest = dict(raw.get("backtest") or {})
    costs = dict(raw.get("costs") or {})

    sig: Dict[str, Any] = {
        "schema": EVALSIG_SCHEMA,
        "data": data,
        "splits": splits,
        "cv": cv,
        "backtest": backtest,
        "costs": costs,
    }

    if data.get("type") == "synthetic" and "seed" in raw:
        try:
            sig["data"]["seed"] = int(raw["seed"])
        except Exception:
            pass

    return sig


def evaluation_hash_from_signature(sig: Dict[str, Any]) -> str:
    return stable_hash(sig)


def evaluation_hash_from_cfg(cfg: Any) -> str:
    return stable_hash(evaluation_signature_from_cfg(cfg))
