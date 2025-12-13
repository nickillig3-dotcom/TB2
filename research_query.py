from __future__ import annotations

import argparse
import sqlite3
from typing import Any, Optional


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def _print_experiments(conn: sqlite3.Connection, dataset_id: Optional[str], limit: int) -> None:
    where = []
    params: list[Any] = []
    if dataset_id:
        where.append("dataset_id = ?")
        params.append(str(dataset_id))

    sql = """
    SELECT experiment_id, created_at, run_name, mode,
           data_type, dataset_id, dataset_version
    FROM experiments
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY experiment_id DESC LIMIT ?"
    params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("No experiments found.")
        return

    print(f"Experiments (limit={limit}{'' if not dataset_id else f', dataset_id={dataset_id}'}):")
    print("id   created_at                mode   run_name              data_type  dataset_id            dataset_version")
    print("-" * 110)
    for r in rows:
        print(
            f"{int(r['experiment_id']):<4d} "
            f"{str(r['created_at']):<24} "
            f"{str(r['mode']):<6} "
            f"{str(r['run_name'])[:20]:<20} "
            f"{str(r['data_type'] or ''):<9} "
            f"{str(r['dataset_id'] or '')[:20]:<20} "
            f"{str(r['dataset_version'] or '')[:20]}"
        )


def _latest_metric(conn: sqlite3.Connection, experiment_id: int, strategy_hash: str, split: str, key: str) -> Optional[float]:
    row = conn.execute(
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
        (int(experiment_id), str(strategy_hash), str(split), str(key)),
    ).fetchone()
    return None if row is None else float(row["value"])


def _leaderboard(
    conn: sqlite3.Connection,
    dataset_id: Optional[str],
    limit: int,
    test_metric_key: str,
    sort_by: str,
) -> None:
    where = []
    params: list[Any] = []
    if dataset_id:
        where.append("e.dataset_id = ?")
        params.append(str(dataset_id))

    sql = """
    SELECT
        e.experiment_id, e.created_at, e.run_name, e.mode,
        e.data_type, e.dataset_id, e.dataset_version,
        b.strategy_hash, b.strategy_name, b.score_key, b.score_value
    FROM experiments e
    JOIN best_candidates b
      ON b.experiment_id = e.experiment_id
     AND b.rank = 1
    """
    if where:
        sql += " WHERE " + " AND ".join(where)

    # fetch a bit more, then sort in python (because test metric is looked up separately)
    sql += " ORDER BY e.experiment_id DESC LIMIT 500"

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("No leaderboard rows found (need best_candidates rank=1).")
        return

    items: list[dict[str, Any]] = []
    for r in rows:
        exp_id = int(r["experiment_id"])
        sh = str(r["strategy_hash"])
        test_val = _latest_metric(conn, exp_id, sh, split="test", key=str(test_metric_key))
        items.append(
            {
                "experiment_id": exp_id,
                "created_at": str(r["created_at"]),
                "run_name": str(r["run_name"]),
                "mode": str(r["mode"]),
                "data_type": str(r["data_type"] or ""),
                "dataset_id": str(r["dataset_id"] or ""),
                "dataset_version": str(r["dataset_version"] or ""),
                "strategy_name": str(r["strategy_name"]),
                "strategy_hash": sh[:12],
                "score_key": str(r["score_key"]),
                "score_value": float(r["score_value"]),
                "test_metric_key": str(test_metric_key),
                "test_metric_value": test_val,
            }
        )

    def _key(it: dict[str, Any]) -> float:
        if sort_by == "test":
            return float(it["test_metric_value"]) if it["test_metric_value"] is not None else float("-inf")
        return float(it["score_value"])

    items.sort(key=_key, reverse=True)
    items = items[: int(limit)]

    print(
        f"Leaderboard (top={limit}, sort_by={sort_by}, test_metric={test_metric_key}"
        f"{'' if not dataset_id else f', dataset_id={dataset_id}'}):"
    )
    print("exp  score_key   score     test_key     test      strategy              hash         dataset_id")
    print("-" * 105)
    for it in items:
        test_s = "None" if it["test_metric_value"] is None else f"{it['test_metric_value']:.4f}"
        print(
            f"{it['experiment_id']:<4d} "
            f"{it['score_key']:<10} "
            f"{it['score_value']:<8.4f} "
            f"{it['test_metric_key']:<11} "
            f"{test_s:<9} "
            f"{it['strategy_name'][:20]:<20} "
            f"{it['strategy_hash']:<12} "
            f"{it['dataset_id'][:20]}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strategy-Miner DB query tool (experiments + leaderboard).")
    p.add_argument("--db", default="strategy_miner.sqlite", help="SQLite DB path (default: strategy_miner.sqlite)")
    p.add_argument("--dataset-id", default=None, help="Filter by dataset_id (optional)")
    p.add_argument("--limit", type=int, default=20, help="Limit rows (default: 20)")

    p.add_argument("--leaderboard", action="store_true", help="Show leaderboard (best candidate per experiment)")
    p.add_argument("--test-metric", default="sharpe_net", help="Metric key to read from test split (default: sharpe_net)")
    p.add_argument("--sort-by", default="cv", choices=["cv", "test"], help="Sort leaderboard by cv or test metric")

    args = p.parse_args(argv)

    conn = _connect(str(args.db))
    try:
        if args.leaderboard:
            _leaderboard(
                conn,
                dataset_id=args.dataset_id,
                limit=int(args.limit),
                test_metric_key=str(args.test_metric),
                sort_by=str(args.sort_by),
            )
        else:
            _print_experiments(conn, dataset_id=args.dataset_id, limit=int(args.limit))
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
