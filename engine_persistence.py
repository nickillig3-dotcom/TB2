from __future__ import annotations

import sqlite3
import zlib
from typing import Any, Dict, Optional

from utils_core import canonical_json, env_fingerprint, now_utc_iso


class SQLitePersistence:
    """Lightweight persistence layer (stdlib sqlite3) for experiment tracking."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._create_tables()
        self._migrate_schema()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _apply_pragmas(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        self.conn.commit()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_name TEXT NOT NULL,
                mode TEXT NOT NULL,
                config_json TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                python TEXT NOT NULL,
                platform TEXT NOT NULL
            );
            """
        )

        # NOTE: fold column is included for NEW DBs.
        # For older DBs we add it in _migrate_schema().
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                split TEXT NOT NULL,
                fold INTEGER,
                strategy_name TEXT NOT NULL,
                strategy_json TEXT NOT NULL,
                strategy_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                elapsed_ms INTEGER,
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment_split ON runs(experiment_id, split);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy_hash ON runs(strategy_hash);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                run_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value REAL,
                PRIMARY KEY(run_id, key),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_key_value ON metrics(key, value);")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                content BLOB NOT NULL,
                content_type TEXT NOT NULL,
                compression TEXT NOT NULL,
                PRIMARY KEY(run_id, name),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS best_candidates (
                experiment_id INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                score_key TEXT NOT NULL,
                score_value REAL NOT NULL,
                strategy_hash TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                strategy_json TEXT NOT NULL,
                PRIMARY KEY(experiment_id, rank),
                FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
            );
            """
        )

        self.conn.commit()

    def _column_exists(self, table: str, column: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        cols = [r["name"] for r in cur.fetchall()]
        return column in cols

    def _migrate_schema(self) -> None:
        """Forward-only tiny migrations for local dev.

        v2: add runs.fold (nullable INTEGER) for walk-forward CV.
        """
        cur = self.conn.cursor()

        if not self._column_exists("runs", "fold"):
            cur.execute("ALTER TABLE runs ADD COLUMN fold INTEGER;")

        # Only create fold index if the column exists (important for migrating older DBs).
        if self._column_exists("runs", "fold"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_fold ON runs(experiment_id, fold);")

        self.conn.commit()

    def create_experiment(
        self,
        run_name: str,
        mode: str,
        config_json: str,
        config_hash: str,
        code_hash: str,
    ) -> int:
        env = env_fingerprint()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO experiments(created_at, run_name, mode, config_json, config_hash, code_hash, python, platform)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (now_utc_iso(), run_name, mode, config_json, config_hash, code_hash, env["python"], env["platform"]),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def start_run(
        self,
        experiment_id: int,
        split: str,
        strategy_name: str,
        strategy_json: str,
        strategy_hash: str,
        fold: Optional[int] = None,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO runs(experiment_id, created_at, split, fold, strategy_name, strategy_json, strategy_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, now_utc_iso(), split, fold, strategy_name, strategy_json, strategy_hash, "running"),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str,
        metrics: Dict[str, float],
        elapsed_ms: int,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE runs
            SET status = ?, error = ?, elapsed_ms = ?
            WHERE run_id = ?
            """,
            (status, error, int(elapsed_ms), int(run_id)),
        )

        if metrics:
            cur.executemany(
                """
                INSERT OR REPLACE INTO metrics(run_id, key, value)
                VALUES (?, ?, ?)
                """,
                [(int(run_id), str(k), float(v)) for k, v in metrics.items()],
            )

        if artifacts:
            for name, obj in artifacts.items():
                payload = canonical_json(obj).encode("utf-8")
                compressed = zlib.compress(payload, level=6)
                cur.execute(
                    """
                    INSERT OR REPLACE INTO artifacts(run_id, name, content, content_type, compression)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (int(run_id), str(name), compressed, "application/json", "zlib"),
                )

        self.conn.commit()

    def store_best_candidates(
        self,
        experiment_id: int,
        ranked: list[dict[str, Any]],
        score_key: str,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM best_candidates WHERE experiment_id = ?", (int(experiment_id),))
        now = now_utc_iso()
        rows = []
        for i, row in enumerate(ranked, start=1):
            rows.append(
                (
                    int(experiment_id),
                    int(i),
                    now,
                    str(score_key),
                    float(row["score_value"]),
                    str(row["strategy_hash"]),
                    str(row["strategy_name"]),
                    str(row["strategy_json"]),
                )
            )
        cur.executemany(
            """
            INSERT INTO best_candidates(experiment_id, rank, created_at, score_key, score_value, strategy_hash, strategy_name, strategy_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def fetch_best_candidates(self, experiment_id: int) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT rank, score_key, score_value, strategy_hash, strategy_name, strategy_json
            FROM best_candidates
            WHERE experiment_id = ?
            ORDER BY rank ASC
            """,
            (int(experiment_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def fetch_top_runs(
        self,
        experiment_id: int,
        split: str,
        metric_key: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT r.run_id, r.strategy_name, r.strategy_hash, r.fold, m.value AS metric
            FROM runs r
            JOIN metrics m ON m.run_id = r.run_id
            WHERE r.experiment_id = ?
              AND r.split = ?
              AND r.status = 'ok'
              AND m.key = ?
            ORDER BY m.value DESC
            LIMIT ?
            """,
            (int(experiment_id), str(split), str(metric_key), int(limit)),
        )
        return [dict(r) for r in cur.fetchall()]

    def fetch_latest_metric_for_strategy(
        self,
        experiment_id: int,
        strategy_hash: str,
        split: str,
        metric_key: str,
    ) -> Optional[float]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT m.value AS value
            FROM runs r
            JOIN metrics m ON m.run_id = r.run_id
            WHERE r.experiment_id = ?
              AND r.strategy_hash = ?
              AND r.split = ?
              AND r.status = 'ok'
              AND m.key = ?
            ORDER BY r.run_id DESC
            LIMIT 1
            """,
            (int(experiment_id), str(strategy_hash), str(split), str(metric_key)),
        )
        row = cur.fetchone()
        return None if row is None else float(row["value"])
