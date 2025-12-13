from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils_core import stable_hash

DATAFP_SCHEMA = "datafp_v1"


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    # best effort
    return dict(getattr(obj, "__dict__", {}) or {})


def file_stat_fingerprint(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def file_sample_fingerprint(path: str, sample_bytes: int = 1024 * 1024) -> Dict[str, Any]:
    """
    Stronger fingerprint (still light): sha256 over head/tail samples + stat.
    Used for meta/audit (NOT necessarily for evaluation_hash).
    """
    base = file_stat_fingerprint(path)
    if not base.get("exists"):
        return base

    p = Path(path)
    size = int(base.get("size", 0))
    try:
        with open(p, "rb") as f:
            head = f.read(sample_bytes)
            head_sha = hashlib.sha256(head).hexdigest()[:16]

            if size > sample_bytes:
                f.seek(max(0, size - sample_bytes))
                tail = f.read(sample_bytes)
                tail_sha = hashlib.sha256(tail).hexdigest()[:16]
            else:
                tail_sha = head_sha
    except Exception as e:
        base["error"] = repr(e)
        return base

    base["sha16_head"] = head_sha
    base["sha16_tail"] = tail_sha
    base["sample_bytes"] = int(sample_bytes)
    return base


def infer_dataset_identity_from_data_dict(data: Dict[str, Any], *, seed: Optional[int] = None) -> Tuple[str, str, str]:
    """
    Returns (dataset_id, dataset_version, method).

    - For synthetic: derived from config+seed
    - For file-based (path): dataset_version derived from file stat unless provided
    - Otherwise: relies on provided dataset_version or "unknown"
    """
    typ = str(data.get("type") or "unknown")

    # user overrides
    dataset_id = data.get("dataset_id") or data.get("path") or data.get("source") or typ
    dataset_id = str(dataset_id)

    provided_ver = data.get("dataset_version")
    if provided_ver:
        return dataset_id, str(provided_ver), "provided"

    if typ == "synthetic":
        # IMPORTANT: this is for identity only; evaluation_signature for synthetic stays unchanged
        base = dict(data)
        if seed is not None:
            base["seed"] = int(seed)
        ver = stable_hash({"schema": "synthetic_identity_v1", **base})
        return dataset_id, ver, "synthetic_config"

    # file-based heuristics
    if "path" in data and data.get("path"):
        st = file_stat_fingerprint(str(data["path"]))
        method = "file_stat" if st.get("exists") else "missing_file"
        ver = stable_hash({"schema": "file_stat_v1", **st})
        return dataset_id, ver, method

    return dataset_id, "unknown", "unknown"


def data_fingerprint_from_cfg_data(data_cfg: Any, *, seed: Optional[int] = None) -> Dict[str, Any]:
    data = _as_dict(data_cfg)
    dataset_id, dataset_version, method = infer_dataset_identity_from_data_dict(data, seed=seed)

    out: Dict[str, Any] = {
        "schema": DATAFP_SCHEMA,
        "type": str(data.get("type") or "unknown"),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,      # this is what should be used inside evaluation_hash
        "version_method": method,
        "eval_identity": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "method": method,
        },
    }

    # Stronger audit for file sources (does NOT change evaluation_hash)
    if data.get("type") != "synthetic" and data.get("path"):
        out["strong_fingerprint"] = file_sample_fingerprint(str(data["path"]))
    else:
        out["strong_fingerprint"] = None

    # add a stable fingerprint hash (for logs/UI)
    tmp = dict(out)
    tmp.pop("fingerprint_hash", None)
    out["fingerprint_hash"] = stable_hash(tmp)

    return out
