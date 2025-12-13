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
    Used for meta/audit. IMPORTANT: This includes mtime_ns for audit, but we will
    NOT use mtime_ns when computing dataset_version.
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

    Design goals:
      - dataset_id should be stable across machines (avoid absolute paths).
      - dataset_version should be CONTENT-based (avoid mtime changes on rewrite/copy).
    """
    typ = str(data.get("type") or "unknown")

    # dataset_id
    if data.get("dataset_id"):
        dataset_id = str(data["dataset_id"])
    elif data.get("path"):
        dataset_id = Path(str(data["path"])).name  # basename only (portable)
    elif data.get("source"):
        dataset_id = str(data["source"])
    else:
        dataset_id = typ

    # dataset_version override
    if data.get("dataset_version"):
        return dataset_id, str(data["dataset_version"]), "provided"

    # synthetic identity
    if typ == "synthetic":
        base = dict(data)
        if seed is not None:
            base["seed"] = int(seed)
        ver = stable_hash({"schema": "synthetic_identity_v1", **base})
        return dataset_id, ver, "synthetic_config"

    # file-based identity (CSV/Parquet/anything with path)
    if data.get("path"):
        fp = file_sample_fingerprint(str(data["path"]))
        if not fp.get("exists"):
            return dataset_id, stable_hash({"schema": "missing_file_v1"}), "missing_file"

        # CONTENT-based version material (NO path, NO mtime_ns)
        material = {
            "size": int(fp.get("size", 0)),
            "sha16_head": str(fp.get("sha16_head", "")),
            "sha16_tail": str(fp.get("sha16_tail", "")),
            "sample_bytes": int(fp.get("sample_bytes", 0)),
        }
        ver = stable_hash({"schema": "file_sample_content_v2", **material})
        return dataset_id, ver, "file_sample_content"

    return dataset_id, "unknown", "unknown"


def data_fingerprint_from_cfg_data(data_cfg: Any, *, seed: Optional[int] = None) -> Dict[str, Any]:
    data = _as_dict(data_cfg)
    dataset_id, dataset_version, method = infer_dataset_identity_from_data_dict(data, seed=seed)

    out: Dict[str, Any] = {
        "schema": DATAFP_SCHEMA,
        "type": str(data.get("type") or "unknown"),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,  # evaluation_hash key uses this
        "version_method": method,
        "eval_identity": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "method": method,
        },
    }

    # Keep strong fingerprint for audit/debug (may contain mtime/path)
    if data.get("type") != "synthetic" and data.get("path"):
        out["strong_fingerprint"] = file_sample_fingerprint(str(data["path"]))
    else:
        out["strong_fingerprint"] = None

    tmp = dict(out)
    tmp.pop("fingerprint_hash", None)
    out["fingerprint_hash"] = stable_hash(tmp)
    return out
