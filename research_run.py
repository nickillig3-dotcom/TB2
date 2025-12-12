from __future__ import annotations

import argparse

from engine_config import load_config
from engine_persistence import SQLitePersistence
from engine_research import run_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strategy-Miner: research runner (CV + holdout + exec + cache stats).")
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--show-top", type=int, default=10, help="Show top-N after run.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    exp_id = run_experiment(cfg)

    db = SQLitePersistence(cfg.persistence.db_path)
    best = db.fetch_best_candidates(exp_id)

    print(f"experiment_id={exp_id} db={cfg.persistence.db_path}")

    exec_stats = db.fetch_latest_artifact_json(
        exp_id,
        split="exec",
        name="exec_stats",
        strategy_hash="__experiment__",
    )

    if isinstance(exec_stats, dict):
        print(
            "Executor:"
            f" mode={exec_stats.get('mode')}"
            f" requested_n_jobs={exec_stats.get('requested_n_jobs')}"
            f" used_n_jobs={exec_stats.get('n_jobs_used')}"
            f" executed_tasks={exec_stats.get('n_tasks_executed')}"
            f" had_fallback={exec_stats.get('had_fallback')}"
        )

        # Cache summary (if present)
        f_total = exec_stats.get("fold_tasks_total")
        f_skip = exec_stats.get("fold_tasks_skipped")
        t_total = exec_stats.get("test_tasks_total")
        t_skip = exec_stats.get("test_tasks_skipped")
        c_gen = exec_stats.get("candidates_generated")
        c_uni = exec_stats.get("candidates_unique")

        if f_total is not None:
            print(f"Cache: fold_skipped={f_skip}/{f_total} | test_skipped={t_skip}/{t_total} | unique_candidates={c_uni}/{c_gen}")

        # Cache diagnostics (why did we compute vs reuse?)
        cache_key = exec_stats.get("cache_key") if isinstance(exec_stats.get("cache_key"), dict) else {}
        eval_h = cache_key.get("evaluation_hash")
        code_h = cache_key.get("code_hash")
        if isinstance(eval_h, str) and isinstance(code_h, str):
            diag = db.cache_diagnostics(eval_h, code_h, exclude_experiment_id=exp_id)
            print(
                "CacheDiag:"
                f" same_eval_experiments={diag.get('same_eval_experiments')}"
                f" same_eval_code_experiments={diag.get('same_eval_code_experiments')}"
                f" latest_same_eval_code_experiment_id={diag.get('latest_same_eval_code_experiment_id')}"
            )
            # Human hint
            if (diag.get("same_eval_experiments", 0) > 0) and (diag.get("same_eval_code_experiments", 0) == 0):
                print("CacheHint: gleiche Evaluation existiert, aber code_hash anders -> Recompute nach Code-Änderung ist erwartbar.")
            elif (diag.get("same_eval_code_experiments", 0) == 0):
                print("CacheHint: cold cache für diese eval_hash+code_hash Kombi (erster Run). Re-run => volle Cache-Hits.")

    else:
        print("Executor: (no exec_stats found)")

    if not best:
        print("No best candidates stored (check run logs).")
        db.close()
        return 0

    print(f"Top-{min(args.show_top, len(best))} (score: {best[0]['score_key']})")
    for row in best[: args.show_top]:
        sh = row["strategy_hash"]
        test_metric = db.fetch_latest_metric_for_strategy(
            exp_id, sh, split="test", metric_key=cfg.research.selection_metric
        )
        test_s = "" if test_metric is None else f" | test_{cfg.research.selection_metric}={test_metric:.4f}"
        print(
            f"#{row['rank']:02d} {row['score_key']}={row['score_value']:.4f} "
            f"strategy={row['strategy_name']} hash={sh[:12]}{test_s}"
        )

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
