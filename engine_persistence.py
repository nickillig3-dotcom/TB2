from __future__ import annotations

import json
import sqlite3
import zlib
from typing import Any, Dict, Optional

from engine_signature import evaluation_hash_from_signature, evaluation_signature_from_config_dict
from utils_core import canonical_json, env_fingerprint, now_utc_iso, stable_hash


class SQLitePersistence:
    """Lightweight persistence layer (stdlib sqlite3) for experiment tracking."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._create_tables()     # create base tables (no risky indexes)
        self._migrate_schema()    # add columns + create indexes safely

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

        # NOTE: CREATE TABLE IF NOT EXISTS will NOT alter existing tables.
        # Therefore: indexes that reference newly-added columns MUST NOT be created here.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_name TEXT NOT NULL,
                mode TEXT NOT NULL,
                config_json TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                evaluation_hash TEXT,
                eval_code_hash TEXT,
                code_hash TEXT NOT NULL,

                -- dataset metadata (indexable)
                data_type TEXT,
                dataset_id TEXT,
                dataset_version TEXT,
                data_fp_hash TEXT,

                python TEXT NOT NULL,
                platform TEXT NOT NULL
            );
            """
        )

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

    def _ensure_indexes(self) -> None:
        """Create indexes only after schema is known to contain required columns."""
        cur = self.conn.cursor()

        # experiments indexes
        if self._column_exists("experiments", "config_hash") and self._column_exists("experiments", "code_hash"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_conf_code ON experiments(config_hash, code_hash);")

        if self._column_exists("experiments", "evaluation_hash") and self._column_exists("experiments", "code_hash"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_eval_code ON experiments(evaluation_hash, code_hash);")

        if self._column_exists("experiments", "evaluation_hash") and self._column_exists("experiments", "eval_code_hash"):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_eval_evalcode ON experiments(evaluation_hash, eval_code_hash);"
            )

        if self._column_exists("experiments", "dataset_id") and self._column_exists("experiments", "dataset_version"):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_id, dataset_version);"
            )

        if self._column_exists("experiments", "dataset_id"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_experiments_dataset_id ON experiments(dataset_id);")

        # runs indexes
        if self._column_exists("runs", "experiment_id") and self._column_exists("runs", "split"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment_split ON runs(experiment_id, split);")

        if self._column_exists("runs", "strategy_hash"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy_hash ON runs(strategy_hash);")

        if self._column_exists("runs", "experiment_id") and self._column_exists("runs", "fold"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_fold ON runs(experiment_id, fold);")

        if (
            self._column_exists("runs", "strategy_hash")
            and self._column_exists("runs", "split")
            and self._column_exists("runs", "fold")
            and self._column_exists("runs", "status")
        ):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_lookup ON runs(strategy_hash, split, fold, status);")

        # metrics indexes
        if self._column_exists("metrics", "key") and self._column_exists("metrics", "value"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_key_value ON metrics(key, value);")

        self.conn.commit()

    def _fetch_meta_data_fingerprint(self, experiment_id: int) -> Optional[dict[str, Any]]:
        """
        Best-effort: read meta/data_fingerprint artifact for an experiment.
        Returns parsed JSON dict or None.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT a.content, a.compression
            FROM runs r
            JOIN artifacts a ON a.run_id = r.run_id
            WHERE r.experiment_id = ?
              AND r.split = 'meta'
              AND r.strategy_hash = '__experiment__'
              AND a.name = 'data_fingerprint'
            ORDER BY r.run_id DESC
            LIMIT 1
            """,
            (int(experiment_id),),
        )
        row = cur.fetchone()
        if row is None:
            return None

        blob = row["content"]
        if row["compression"] == "zlib":
            blob = zlib.decompress(blob)

        try:
            obj = json.loads(blob.decode("utf-8"))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _migrate_schema(self) -> None:
        """Forward-only tiny migrations for local dev."""
        cur = self.conn.cursor()

        # v2: runs.fold
        if not self._column_exists("runs", "fold"):
            cur.execute("ALTER TABLE runs ADD COLUMN fold INTEGER;")

        # v3: experiments.evaluation_hash
        if not self._column_exists("experiments", "evaluation_hash"):
            cur.execute("ALTER TABLE experiments ADD COLUMN evaluation_hash TEXT;")

        # v4: experiments.eval_code_hash
        if not self._column_exists("experiments", "eval_code_hash"):
            cur.execute("ALTER TABLE experiments ADD COLUMN eval_code_hash TEXT;")

        # v5: dataset metadata columns
        if not self._column_exists("experiments", "data_type"):
            cur.execute("ALTER TABLE experiments ADD COLUMN data_type TEXT;")
        if not self._column_exists("experiments", "dataset_id"):
            cur.execute("ALTER TABLE experiments ADD COLUMN dataset_id TEXT;")
        if not self._column_exists("experiments", "dataset_version"):
            cur.execute("ALTER TABLE experiments ADD COLUMN dataset_version TEXT;")
        if not self._column_exists("experiments", "data_fp_hash"):
            cur.execute("ALTER TABLE experiments ADD COLUMN data_fp_hash TEXT;")

        self.conn.commit()

        # Backfill evaluation_hash if missing (best effort)
        if self._column_exists("experiments", "evaluation_hash"):
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT experiment_id, config_json
                FROM experiments
                WHERE evaluation_hash IS NULL OR evaluation_hash = ''
                """
            )
            rows = cur.fetchall()
            if rows:
                for r in rows:
                    try:
                        raw = json.loads(r["config_json"])
                        sig = evaluation_signature_from_config_dict(raw)
                        eh = evaluation_hash_from_signature(sig)
                    except Exception:
                        continue
                    cur.execute(
                        "UPDATE experiments SET evaluation_hash = ? WHERE experiment_id = ?",
                        (str(eh), int(r["experiment_id"])),
                    )
                self.conn.commit()

        # Backfill eval_code_hash (safe fallback: set to code_hash if missing)
        if self._column_exists("experiments", "eval_code_hash") and self._column_exists("experiments", "code_hash"):
            cur = self.conn.cursor()
            cur.execute(
                """
                UPDATE experiments
                SET eval_code_hash = code_hash
                WHERE eval_code_hash IS NULL OR eval_code_hash = ''
                """
            )
            self.conn.commit()

        # Backfill dataset metadata (prefer meta artifact; fallback to config_json best-effort)
        if (
            self._column_exists("experiments", "dataset_id")
            and self._column_exists("experiments", "dataset_version")
            and self._column_exists("experiments", "data_type")
            and self._column_exists("experiments", "data_fp_hash")
        ):
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT experiment_id, config_json, data_type, dataset_id, dataset_version, data_fp_hash
                FROM experiments
                WHERE dataset_id IS NULL OR dataset_id = ''
                   OR dataset_version IS NULL OR dataset_version = ''
                   OR data_type IS NULL OR data_type = ''
                   OR data_fp_hash IS NULL OR data_fp_hash = ''
                """
            )
            rows = cur.fetchall()
            if rows:
                for r in rows:
                    exp_id = int(r["experiment_id"])
                    fp = self._fetch_meta_data_fingerprint(exp_id)

                    if isinstance(fp, dict):
                        data_type = fp.get("type") or "unknown"
                        dataset_id = fp.get("dataset_id") or "unknown"
                        dataset_version = fp.get("dataset_version") or "unknown"
                        data_fp_hash = fp.get("fingerprint_hash") or stable_hash(fp)
                    else:
                        # fallback: parse config_json
                        try:
                            raw = json.loads(r["config_json"])
                            data = dict(raw.get("data") or {})
                            data_type = str(data.get("type") or "unknown")

                            # If user provided dataset fields in config, use them
                            dataset_id = str(data.get("dataset_id") or (data.get("path") or data_type))
                            dataset_version = str(data.get("dataset_version") or "unknown")

                            # best-effort stable hash for identifying that we backfilled
                            data_fp_hash = stable_hash(
                                {
                                    "schema": "datafp_backfill_v1",
                                    "type": data_type,
                                    "dataset_id": dataset_id,
                                    "dataset_version": dataset_version,
                                }
                            )
                        except Exception:
                            continue

                    cur.execute(
                        """
                        UPDATE experiments
                        SET data_type = ?, dataset_id = ?, dataset_version = ?, data_fp_hash = ?
                        WHERE experiment_id = ?
                        """,
                        (str(data_type), str(dataset_id), str(dataset_version), str(data_fp_hash), exp_id),
                    )
                self.conn.commit()

        # Finally: create indexes safely
        self._ensure_indexes()

    def create_experiment(
        self,
        run_name: str,
        mode: str,
        config_json: str,
        config_hash: str,
        evaluation_hash: str,
        eval_code_hash: str,
        code_hash: str,
        data_type: str,
        dataset_id: str,
        dataset_version: str,
        data_fp_hash: str,
    ) -> int:
        env = env_fingerprint()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO experiments(
                created_at, run_name, mode, config_json, config_hash,
                evaluation_hash, eval_code_hash, code_hash,
                data_type, dataset_id, dataset_version, data_fp_hash,
                python, platform
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc_iso(),
                run_name,
                mode,
                config_json,
                config_hash,
                evaluation_hash,
                eval_code_hash,
                code_hash,
                data_type,
                dataset_id,
                dataset_version,
                data_fp_hash,
                env["python"],
                env["platform"],
            ),
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
        if row is None:
            return None

        v = row["value"]
        if v is None:
            return None

        try:
            return float(v)
        except (TypeError, ValueError):
            return None


    def fetch_latest_artifact_json(
        self,
        experiment_id: int,
        split: str,
        name: str,
        strategy_hash: Optional[str] = None,
    ) -> Optional[Any]:
        cur = self.conn.cursor()
        if strategy_hash is None:
            cur.execute(
                """
                SELECT a.content, a.compression
                FROM runs r
                JOIN artifacts a ON a.run_id = r.run_id
                WHERE r.experiment_id = ?
                  AND r.split = ?
                  AND a.name = ?
                ORDER BY r.run_id DESC
                LIMIT 1
                """,
                (int(experiment_id), str(split), str(name)),
            )
        else:
            cur.execute(
                """
                SELECT a.content, a.compression
                FROM runs r
                JOIN artifacts a ON a.run_id = r.run_id
                WHERE r.experiment_id = ?
                  AND r.split = ?
                  AND r.strategy_hash = ?
                  AND a.name = ?
                ORDER BY r.run_id DESC
                LIMIT 1
                """,
                (int(experiment_id), str(split), str(strategy_hash), str(name)),
            )

        row = cur.fetchone()
        if row is None:
            return None

        blob = row["content"]
        if row["compression"] == "zlib":
            blob = zlib.decompress(blob)

        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            return None

    # -------- Cache helpers ---------------------------------------------------

    def fetch_run_metrics(self, run_id: int) -> Dict[str, float]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT key, value
            FROM metrics
            WHERE run_id = ?
            """,
            (int(run_id),),
        )
        out: Dict[str, float] = {}
        for r in cur.fetchall():
            out[str(r["key"])] = float(r["value"]) if r["value"] is not None else float("nan")
        return out

    def fetch_artifact_json_by_run_id(self, run_id: int, name: str) -> Optional[Any]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT content, compression
            FROM artifacts
            WHERE run_id = ?
              AND name = ?
            LIMIT 1
            """,
            (int(run_id), str(name)),
        )
        row = cur.fetchone()
        if row is None:
            return None

        blob = row["content"]
        if row["compression"] == "zlib":
            blob = zlib.decompress(blob)

        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            return None

    def fetch_cached_run_ref(
        self,
        evaluation_hash: str,
        eval_code_hash: str,
        strategy_hash: str,
        split: str,
        fold: Optional[int],
        exclude_experiment_id: int,
    ) -> Optional[dict[str, Any]]:
        """
        Find a previous run (different experiment) with same (evaluation_hash, eval_code_hash, strategy_hash, split, fold).
        Returns: {"run_id": ..., "experiment_id": ...} or None.
        """
        cur = self.conn.cursor()

        if fold is None:
            cur.execute(
                """
                SELECT r.run_id, r.experiment_id
                FROM runs r
                JOIN experiments e ON e.experiment_id = r.experiment_id
                WHERE e.evaluation_hash = ?
                  AND e.eval_code_hash = ?
                  AND r.strategy_hash = ?
                  AND r.split = ?
                  AND r.fold IS NULL
                  AND r.status = 'ok'
                  AND r.experiment_id != ?
                ORDER BY r.run_id DESC
                LIMIT 1
                """,
                (str(evaluation_hash), str(eval_code_hash), str(strategy_hash), str(split), int(exclude_experiment_id)),
            )
        else:
            cur.execute(
                """
                SELECT r.run_id, r.experiment_id
                FROM runs r
                JOIN experiments e ON e.experiment_id = r.experiment_id
                WHERE e.evaluation_hash = ?
                  AND e.eval_code_hash = ?
                  AND r.strategy_hash = ?
                  AND r.split = ?
                  AND r.fold = ?
                  AND r.status = 'ok'
                  AND r.experiment_id != ?
                ORDER BY r.run_id DESC
                LIMIT 1
                """,
                (str(evaluation_hash), str(eval_code_hash), str(strategy_hash), str(split), int(fold), int(exclude_experiment_id)),
            )

        row = cur.fetchone()
        return None if row is None else {"run_id": int(row["run_id"]), "experiment_id": int(row["experiment_id"])}

    def cache_diagnostics(
        self,
        evaluation_hash: str,
        eval_code_hash: str,
        code_hash: str,
        exclude_experiment_id: int,
    ) -> dict[str, Any]:
        """
        Explain cache hits/misses:
          - same_eval_experiments: same evaluation_hash exists at all
          - same_eval_evalcode_experiments: same evaluation_hash AND eval_code_hash (actual cache key)
          - same_eval_fullcode_experiments: same evaluation_hash AND full code_hash
        """
        cur = self.conn.cursor()

        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM experiments
            WHERE evaluation_hash = ?
              AND experiment_id != ?
            """,
            (str(evaluation_hash), int(exclude_experiment_id)),
        )
        same_eval = int(cur.fetchone()["n"])

        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM experiments
            WHERE evaluation_hash = ?
              AND eval_code_hash = ?
              AND experiment_id != ?
            """,
            (str(evaluation_hash), str(eval_code_hash), int(exclude_experiment_id)),
        )
        same_eval_evalcode = int(cur.fetchone()["n"])

        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM experiments
            WHERE evaluation_hash = ?
              AND code_hash = ?
              AND experiment_id != ?
            """,
            (str(evaluation_hash), str(code_hash), int(exclude_experiment_id)),
        )
        same_eval_fullcode = int(cur.fetchone()["n"])

        cur.execute(
            """
            SELECT MAX(experiment_id) AS max_id
            FROM experiments
            WHERE evaluation_hash = ?
              AND eval_code_hash = ?
              AND experiment_id != ?
            """,
            (str(evaluation_hash), str(eval_code_hash), int(exclude_experiment_id)),
        )
        row = cur.fetchone()
        latest_same_eval_evalcode = None if row is None or row["max_id"] is None else int(row["max_id"])

        return {
            "same_eval_experiments": same_eval,
            "same_eval_evalcode_experiments": same_eval_evalcode,
            "same_eval_fullcode_experiments": same_eval_fullcode,
            "latest_same_eval_evalcode_experiment_id": latest_same_eval_evalcode,
        }
