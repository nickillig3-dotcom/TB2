from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import uuid
import csv

from engine_config import StrategyMinerConfig, ResultsConfig


logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """
    JSON serializer helper that handles numpy types and dataclasses-ish objects.
    """
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class ResultStore:
    """
    Stores StrategyMiner runs and strategy results.

    Storage:
      - SQLite (default) for scalable persistence (single file).
      - Optional CSV export (flattened columns + JSON strings for nested dicts).

    Schema design goals:
      - stable and append-only
      - easy to query (top results, per-run analysis)
      - future-proof: nested dicts stored as JSON
    """
    def __init__(self, cfg: ResultsConfig):
        self.cfg = cfg
        self.sqlite_path = Path(cfg.sqlite_path).expanduser().resolve()
        self.csv_path = Path(cfg.csv_path).expanduser().resolve() if cfg.csv_path else None

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.sqlite_path))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    engine_mode TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    data_rows INTEGER,
                    data_cols INTEGER,
                    data_start TEXT,
                    data_end TEXT,
                    notes TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    strategy_name TEXT NOT NULL,
                    rank_score REAL NOT NULL,
                    robustness_score REAL NOT NULL,
                    strategy_json TEXT NOT NULL,
                    train_metrics_json TEXT NOT NULL,
                    test_metrics_json TEXT NOT NULL,
                    robustness_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_strategy_results_run_id ON strategy_results(run_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_strategy_results_rank_score ON strategy_results(rank_score)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_strategy_results_robustness_score ON strategy_results(robustness_score)")
        logger.info("ResultStore initialized: %s", self.sqlite_path)

    def start_run(self, cfg: StrategyMinerConfig, data_meta: Optional[Dict[str, Any]] = None, notes: str = "") -> str:
        """
        Creates a run entry and returns a run_id.
        """
        self.init_db()

        run_id = uuid.uuid4().hex
        meta = data_meta or {}
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO runs (run_id, created_at, engine_mode, config_json, data_rows, data_cols, data_start, data_end, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now_utc_iso(),
                    cfg.engine.mode,
                    cfg.to_json(indent=None),
                    int(meta.get("rows")) if meta.get("rows") is not None else None,
                    int(meta.get("cols")) if meta.get("cols") is not None else None,
                    str(meta.get("start")) if meta.get("start") is not None else None,
                    str(meta.get("end")) if meta.get("end") is not None else None,
                    notes,
                ),
            )
        return run_id

    def save_results(self, run_id: str, results: Sequence[Dict[str, Any]]) -> None:
        """
        Persists top results. Each result is a dict returned by StrategyMiner.mine().
        """
        if not results:
            logger.warning("No results to save for run_id=%s", run_id)
            return

        with self._connect() as con:
            for rank, r in enumerate(results, start=1):
                robustness = r.get("robustness", {}) or {}
                con.execute(
                    """
                    INSERT INTO strategy_results
                      (run_id, created_at, rank, strategy_name, rank_score, robustness_score,
                       strategy_json, train_metrics_json, test_metrics_json, robustness_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        now_utc_iso(),
                        int(rank),
                        str(r.get("strategy_name")),
                        float(r.get("rank_score", 0.0) or 0.0),
                        float(robustness.get("robustness_score", 0.0) or 0.0),
                        dumps(r.get("strategy_config", {})),
                        dumps(r.get("train_metrics", {})),
                        dumps(r.get("test_metrics", {})),
                        dumps(robustness),
                    ),
                )

        logger.info("Saved %s results into %s (run_id=%s)", len(results), self.sqlite_path, run_id)

        # Optional: export CSV snapshot for quick inspection
        if self.csv_path:
            try:
                self.export_run_to_csv(run_id, self.csv_path)
            except Exception as e:
                logger.warning("CSV export failed: %s", e)

    def export_run_to_csv(self, run_id: str, csv_path: Path) -> None:
        """
        Exports all strategies for a run into a CSV file.

        The CSV is "flat": nested dicts are stored as JSON strings in columns:
          strategy_json, train_metrics_json, test_metrics_json, robustness_json
        """
        rows = self.load_results_for_run(run_id, limit=None, order_by="rank_score")
        if not rows:
            logger.warning("No rows for run_id=%s to export", run_id)
            return

        csv_path = Path(csv_path).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        logger.info("Exported run_id=%s to CSV: %s", run_id, csv_path)

    def load_results_for_run(
        self,
        run_id: str,
        limit: Optional[int] = 50,
        order_by: str = "rank_score",
    ) -> List[Dict[str, Any]]:
        """
        Load results for a specific run.
        """
        if order_by not in ("rank_score", "robustness_score"):
            raise ValueError("order_by must be 'rank_score' or 'robustness_score'")

        limit_sql = "" if limit is None else "LIMIT ?"
        q = f"""
            SELECT run_id, created_at, rank, strategy_name, rank_score, robustness_score,
                   strategy_json, train_metrics_json, test_metrics_json, robustness_json
            FROM strategy_results
            WHERE run_id = ?
            ORDER BY {order_by} DESC
            {limit_sql}
        """
        params = (run_id,) if limit is None else (run_id, int(limit))

        out: List[Dict[str, Any]] = []
        with self._connect() as con:
            for row in con.execute(q, params):
                out.append(
                    {
                        "run_id": row[0],
                        "created_at": row[1],
                        "rank": row[2],
                        "strategy_name": row[3],
                        "rank_score": row[4],
                        "robustness_score": row[5],
                        "strategy_json": row[6],
                        "train_metrics_json": row[7],
                        "test_metrics_json": row[8],
                        "robustness_json": row[9],
                    }
                )
        return out

    def load_top_results(self, limit: int = 10, order_by: str = "rank_score") -> List[Dict[str, Any]]:
        """
        Load top results across all runs (merged leaderboard).
        """
        if order_by not in ("rank_score", "robustness_score"):
            raise ValueError("order_by must be 'rank_score' or 'robustness_score'")

        q = f"""
            SELECT run_id, created_at, rank, strategy_name, rank_score, robustness_score,
                   strategy_json, train_metrics_json, test_metrics_json, robustness_json
            FROM strategy_results
            ORDER BY {order_by} DESC
            LIMIT ?
        """
        out: List[Dict[str, Any]] = []
        with self._connect() as con:
            for row in con.execute(q, (int(limit),)):
                out.append(
                    {
                        "run_id": row[0],
                        "created_at": row[1],
                        "rank": row[2],
                        "strategy_name": row[3],
                        "rank_score": row[4],
                        "robustness_score": row[5],
                        "strategy_config": json.loads(row[6]),
                        "train_metrics": json.loads(row[7]),
                        "test_metrics": json.loads(row[8]),
                        "robustness": json.loads(row[9]),
                    }
                )
        return out
