from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from statistics import NormalDist
from typing import Any, Optional

_NORM = NormalDist()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _norm_cdf(z: float) -> float:
    try:
        return float(_NORM.cdf(float(z)))
    except Exception:
        return float("nan")


def _norm_ppf(p: float) -> float:
    try:
        pp = float(p)
    except Exception:
        return float("nan")
    if not (pp == pp):  # NaN
        return float("nan")
    eps = 1e-12
    if pp < eps:
        pp = eps
    if pp > 1.0 - eps:
        pp = 1.0 - eps
    return float(_NORM.inv_cdf(pp))


def _sharpe_se(sr_per: float, n_obs: int, skew: float, kurt: float) -> float:
    """
    Approx Sharpe SE with skew/kurt adjustment:
      se ≈ sqrt((1 - skew*SR + ((kurt-1)/4)*SR^2) / (n-1))
    Defaults if skew/kurt missing: skew=0, kurt=3.
    """
    n = int(n_obs)
    sr = float(sr_per)
    if n <= 1 or not (sr == sr):
        return float("nan")

    sk = float(skew) if (skew == skew) else 0.0
    kt = float(kurt) if (kurt == kurt) else 3.0

    term = 1.0 - sk * sr + ((kt - 1.0) / 4.0) * (sr ** 2)
    if (not (term == term)) or term <= 0.0:
        return float("nan")
    return (term / float(n - 1)) ** 0.5


def _dsr_prob(sr_per: float, n_obs: int, skew: float, kurt: float, trials: int, sr0_per: float = 0.0) -> float:
    """
    Deflated Sharpe Ratio probability using Blom approximation:
      p = (M - 0.375)/(M + 0.25)
      z_M = Phi^{-1}(p)
      SR* = SR0 + se*z_M
      DSR = Phi((SR_hat - SR*)/se)
    """
    m = int(trials)
    if m < 1:
        m = 1

    se = _sharpe_se(sr_per, n_obs, skew, kurt)
    if (not (se == se)) or se <= 0.0:
        return float("nan")

    if m == 1:
        sr_star = float(sr0_per)
    else:
        p = (float(m) - 0.375) / (float(m) + 0.25)
        z_m = _norm_ppf(p)
        sr_star = float(sr0_per) + se * z_m

    z = (float(sr_per) - sr_star) / se
    return _norm_cdf(z)


def _latest_metric(
    conn: sqlite3.Connection,
    experiment_id: int,
    strategy_hash: str,
    split: str,
    key: str,
    *,
    fold_is_null: bool = True,
) -> Optional[float]:
    """
    Latest metric for (experiment_id, strategy_hash, split, key).
    For cv/test we want fold IS NULL.
    """
    if fold_is_null:
        row = conn.execute(
            """
            SELECT m.value AS value
            FROM runs r
            JOIN metrics m ON m.run_id = r.run_id
            WHERE r.experiment_id = ?
              AND r.strategy_hash = ?
              AND r.split = ?
              AND r.fold IS NULL
              AND r.status = 'ok'
              AND m.key = ?
            ORDER BY r.run_id DESC
            LIMIT 1
            """,
            (int(experiment_id), str(strategy_hash), str(split), str(key)),
        ).fetchone()
    else:
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


def _latest_artifact_json(
    conn: sqlite3.Connection,
    experiment_id: int,
    split: str,
    name: str,
    strategy_hash: str,
) -> Optional[Any]:
    row = conn.execute(
        """
        SELECT a.content AS content, a.compression AS compression
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
    ).fetchone()

    if row is None:
        return None

    blob = row["content"]
    comp = str(row["compression"] or "")
    if comp == "zlib":
        try:
            blob = zlib.decompress(blob)
        except Exception:
            return None
    try:
        return json.loads(blob.decode("utf-8"))
    except Exception:
        return None


def _parse_selection_from_config(config_json: str) -> tuple[str, str]:
    """
    Returns (selection_metric, cv_agg). Defaults to ('sharpe_net','median') if missing.
    """
    try:
        raw = json.loads(config_json)
        r = raw.get("research") or {}
        sel = str(r.get("selection_metric") or "sharpe_net")
        agg = str(r.get("cv_agg") or "median")
        return sel, agg
    except Exception:
        return "sharpe_net", "median"


def _print_experiments(conn: sqlite3.Connection, dataset_id: Optional[str], limit: int, with_config: bool) -> None:
    where = []
    params: list[Any] = []
    if dataset_id:
        where.append("dataset_id = ?")
        params.append(str(dataset_id))

    sql = """
    SELECT experiment_id, created_at, run_name, mode,
           data_type, dataset_id, dataset_version, config_json
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

    if with_config:
        print(f"Experiments (limit={limit}{'' if not dataset_id else f', dataset_id={dataset_id}'}, with_config=True):")
        print("id   created_at                mode   run_name              data_type  dataset_id            dataset_version        selection_metric cv_agg")
        print("-" * 130)
        for r in rows:
            sel, agg = _parse_selection_from_config(str(r["config_json"]))
            print(
                f"{int(r['experiment_id']):<4d} "
                f"{str(r['created_at']):<24} "
                f"{str(r['mode']):<6} "
                f"{str(r['run_name'])[:20]:<20} "
                f"{str(r['data_type'] or ''):<9} "
                f"{str(r['dataset_id'] or '')[:20]:<20} "
                f"{str(r['dataset_version'] or '')[:20]:<20} "
                f"{sel[:15]:<15} "
                f"{agg:<5}"
            )
    else:
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


def _leaderboard(
    conn: sqlite3.Connection,
    dataset_id: Optional[str],
    limit: int,
    sort_by: str,
    score_key_filter: Optional[str],
) -> None:
    where = []
    params: list[Any] = []

    if dataset_id:
        where.append("e.dataset_id = ?")
        params.append(str(dataset_id))

    if score_key_filter:
        where.append("b.score_key = ?")
        params.append(str(score_key_filter))

    sql = """
    SELECT
        e.experiment_id, e.created_at, e.run_name, e.mode,
        e.data_type, e.dataset_id, e.dataset_version,
        e.config_json,
        b.strategy_hash, b.strategy_name, b.score_key, b.score_value
    FROM experiments e
    JOIN best_candidates b
      ON b.experiment_id = e.experiment_id
     AND b.rank = 1
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY e.experiment_id DESC LIMIT 500"

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("No leaderboard rows found (need best_candidates rank=1).")
        return

    items: list[dict[str, Any]] = []
    for r in rows:
        exp_id = int(r["experiment_id"])
        sh_full = str(r["strategy_hash"])
        sh = sh_full[:12]

        sel_metric, cv_agg = _parse_selection_from_config(str(r["config_json"]))
        cv_sel_key = f"{sel_metric}_valid_{cv_agg}"

        cv_sel = _latest_metric(conn, exp_id, sh_full, "cv", cv_sel_key, fold_is_null=True)
        score_value = float(r["score_value"])
        score_key = str(r["score_key"])

        test_sharpe = _latest_metric(conn, exp_id, sh_full, "test", "sharpe_net", fold_is_null=True)

        # Trials for DSR: prefer cv metric dsr_trials; fallback to exec_stats candidates_unique; else 1.
        dsr_trials = _latest_metric(conn, exp_id, sh_full, "cv", "dsr_trials", fold_is_null=True)
        trials = 1
        if dsr_trials is not None:
            trials = max(1, int(dsr_trials))
        else:
            exec_stats = _latest_artifact_json(conn, exp_id, "exec", "exec_stats", "__experiment__")
            if isinstance(exec_stats, dict):
                cu = exec_stats.get("candidates_unique")
                try:
                    trials = max(1, int(cu))
                except Exception:
                    trials = 1

        # Test selection metric value
        test_sel: Optional[float] = None
        if sel_metric == "dsr_net":
            sr_per = _latest_metric(conn, exp_id, sh_full, "test", "sharpe_net_per", fold_is_null=True)
            if sr_per is None:
                # derive from annualized sharpe + ppy if needed
                sharpe_ann = _latest_metric(conn, exp_id, sh_full, "test", "sharpe_net", fold_is_null=True)
                ppy = _latest_metric(conn, exp_id, sh_full, "test", "periods_per_year", fold_is_null=True)
                if ppy is None or not (ppy == ppy):
                    ppy = 252.0
                if sharpe_ann is not None and ppy is not None and ppy > 0:
                    sr_per = float(sharpe_ann / (ppy ** 0.5))

            n_obs = _latest_metric(conn, exp_id, sh_full, "test", "n_obs", fold_is_null=True)
            skew = _latest_metric(conn, exp_id, sh_full, "test", "skew_net", fold_is_null=True)
            kurt = _latest_metric(conn, exp_id, sh_full, "test", "kurt_net", fold_is_null=True)

            if sr_per is not None and n_obs is not None:
                sk = float(skew) if skew is not None else 0.0
                kt = float(kurt) if kurt is not None else 3.0
                test_sel = _dsr_prob(float(sr_per), int(n_obs), sk, kt, trials=trials, sr0_per=0.0)
        else:
            test_sel = _latest_metric(conn, exp_id, sh_full, "test", sel_metric, fold_is_null=True)

        items.append(
            {
                "experiment_id": exp_id,
                "mode": str(r["mode"]),
                "run_name": str(r["run_name"]),
                "dataset_id": str(r["dataset_id"] or ""),
                "dataset_version": str(r["dataset_version"] or ""),
                "strategy_name": str(r["strategy_name"]),
                "strategy_hash": sh,
                "score_key": score_key,
                "score_value": score_value,
                "selection_metric": sel_metric,
                "cv_agg": cv_agg,
                "cv_sel_key": cv_sel_key,
                "cv_sel_value": cv_sel,
                "test_sel_value": test_sel,
                "test_sharpe_net": test_sharpe,
                "dsr_trials": trials,
            }
        )

    def _sort_key(it: dict[str, Any]) -> float:
        if sort_by == "test":
            v = it["test_sel_value"]
            return float(v) if v is not None and (v == v) else float("-inf")
        if sort_by == "test_sharpe":
            v = it["test_sharpe_net"]
            return float(v) if v is not None and (v == v) else float("-inf")
        return float(it["score_value"])

    items.sort(key=_sort_key, reverse=True)
    items = items[: int(limit)]

    skf = score_key_filter if score_key_filter else "any"
    print(f"Leaderboard (top={limit}, sort_by={sort_by}, score_key={skf}{'' if not dataset_id else f', dataset_id={dataset_id}'}):")
    print("exp  score_key   score     sel_metric   cv_sel                test_sel     test_sharpe  trials  strategy              hash         dataset_id")
    print("-" * 140)

    for it in items:
        cv_sel_s = "None" if it["cv_sel_value"] is None else f"{float(it['cv_sel_value']):.4f}"
        test_sel_s = "None" if it["test_sel_value"] is None else f"{float(it['test_sel_value']):.4f}"
        test_sh_s = "None" if it["test_sharpe_net"] is None else f"{float(it['test_sharpe_net']):.4f}"
        print(
            f"{it['experiment_id']:<4d} "
            f"{it['score_key']:<10} "
            f"{it['score_value']:<8.4f} "
            f"{it['selection_metric']:<11} "
            f"{it['cv_sel_key'][:18]:<18}={cv_sel_s:<8} "
            f"{test_sel_s:<11} "
            f"{test_sh_s:<11} "
            f"{int(it['dsr_trials']):<6d} "
            f"{it['strategy_name'][:20]:<20} "
            f"{it['strategy_hash']:<12} "
            f"{it['dataset_id'][:20]}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strategy-Miner DB query tool (experiments + selection-aware leaderboard).")
    p.add_argument("--db", default="strategy_miner.sqlite", help="SQLite DB path (default: strategy_miner.sqlite)")
    p.add_argument("--dataset-id", default=None, help="Filter by dataset_id (optional)")
    p.add_argument("--limit", type=int, default=20, help="Limit rows (default: 20)")

    p.add_argument("--with-config", action="store_true", help="Show selection_metric/cv_agg parsed from config_json")

    p.add_argument("--leaderboard", action="store_true", help="Show leaderboard (best candidate per experiment)")
    p.add_argument("--sort-by", default="cv", choices=["cv", "test", "test_sharpe"], help="Sort leaderboard by cv score, test selection metric, or test sharpe")

    p.add_argument(
        "--score-key",
        default="cv_score",
        help="Filter leaderboard by best_candidates.score_key (default: cv_score). Use 'any' to disable.",
    )

    args = p.parse_args(argv)

    score_key_filter: Optional[str] = None
    if isinstance(args.score_key, str):
        s = args.score_key.strip().lower()
        if s not in {"any", "*", ""}:
            score_key_filter = args.score_key.strip()

    conn = _connect(str(args.db))
    try:
        if args.leaderboard:
            _leaderboard(
                conn,
                dataset_id=args.dataset_id,
                limit=int(args.limit),
                sort_by=str(args.sort_by),
                score_key_filter=score_key_filter,
            )
        else:
            _print_experiments(
                conn,
                dataset_id=args.dataset_id,
                limit=int(args.limit),
                with_config=bool(args.with_config),
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
