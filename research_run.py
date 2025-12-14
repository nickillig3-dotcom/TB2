from __future__ import annotations

import argparse

from engine_config import load_config
from engine_metrics import deflated_sharpe_ratio
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

    data_fp = db.fetch_latest_artifact_json(
        exp_id, split="meta", name="data_fingerprint", strategy_hash="__experiment__"
    )
    if isinstance(data_fp, dict):
        print(
            "Data:"
            f" type={data_fp.get('type')}"
            f" id={data_fp.get('dataset_id')}"
            f" version={data_fp.get('dataset_version')}"
            f" method={data_fp.get('version_method')}"
        )

    exec_stats = db.fetch_latest_artifact_json(
        exp_id, split="exec", name="exec_stats", strategy_hash="__experiment__"
    )

    c_uni = None
    if isinstance(exec_stats, dict):
        print(
            "Executor:"
            f" mode={exec_stats.get('mode')}"
            f" requested_n_jobs={exec_stats.get('requested_n_jobs')}"
            f" used_n_jobs={exec_stats.get('n_jobs_used')}"
            f" executed_tasks={exec_stats.get('n_tasks_executed')}"
            f" had_fallback={exec_stats.get('had_fallback')}"
        )
        f_total = exec_stats.get("fold_tasks_total")
        f_skip = exec_stats.get("fold_tasks_skipped")
        t_total = exec_stats.get("test_tasks_total")
        t_skip = exec_stats.get("test_tasks_skipped")
        c_gen = exec_stats.get("candidates_generated")
        c_uni = exec_stats.get("candidates_unique")
        if f_total is not None:
            print(f"Cache: fold_skipped={f_skip}/{f_total} | test_skipped={t_skip}/{t_total} | unique_candidates={c_uni}/{c_gen}")

        cache_key = exec_stats.get("cache_key") if isinstance(exec_stats.get("cache_key"), dict) else {}
        eval_h = cache_key.get("evaluation_hash")
        eval_code_h = cache_key.get("eval_code_hash")
        code_h = cache_key.get("code_hash")
        if isinstance(eval_h, str) and isinstance(eval_code_h, str) and isinstance(code_h, str):
            diag = db.cache_diagnostics(eval_h, eval_code_h, code_h, exclude_experiment_id=exp_id)
            print(
                "CacheDiag:"
                f" same_eval_experiments={diag.get('same_eval_experiments')}"
                f" same_eval_evalcode_experiments={diag.get('same_eval_evalcode_experiments')}"
                f" same_eval_fullcode_experiments={diag.get('same_eval_fullcode_experiments')}"
                f" latest_same_eval_evalcode_experiment_id={diag.get('latest_same_eval_evalcode_experiment_id')}"
            )
            if (diag.get("same_eval_experiments", 0) > 0) and (diag.get("same_eval_evalcode_experiments", 0) == 0):
                print("CacheHint: gleiche Evaluation existiert, aber eval_code_hash anders -> Recompute ist erwartbar (Eval-Engine geändert / erster Run nach eval_code_key).")
            elif (diag.get("same_eval_evalcode_experiments", 0) == 0):
                print("CacheHint: cold cache für eval_hash+eval_code_hash (erster Run). Re-run => volle Cache-Hits.")
            elif (diag.get("same_eval_evalcode_experiments", 0) > 0) and (diag.get("same_eval_fullcode_experiments", 0) == 0):
                print("CacheHint: eval_code gleich, full code anders -> Cache sollte trotzdem greifen (genau dafür ist eval_code_hash da).")
    else:
        print("Executor: (no exec_stats found)")

    if not best:
        print("No best candidates stored (check run logs).")
        db.close()
        return 0

    top_n = min(args.show_top, len(best))
    print(f"Top-{top_n} (score: {best[0]['score_key']})")

    sel_metric = cfg.research.selection_metric
    cv_key = f"{sel_metric}_valid_{cfg.research.cv_agg}"
    trials = int(c_uni) if isinstance(c_uni, (int, float)) and float(c_uni) > 0 else 1

    for row in best[: args.show_top]:
        sh = row["strategy_hash"]

        # Show the actual selection metric aggregate stored in CV split metrics
        cv_sel = db.fetch_latest_metric_for_strategy(exp_id, sh, split="cv", metric_key=cv_key)

        # Overfitting diagnostics (train vs valid on the same selection metric)
        gap_key = f"{sel_metric}_train_valid_gap_{cfg.research.cv_agg}"
        cv_gap = db.fetch_latest_metric_for_strategy(exp_id, sh, split="cv", metric_key=gap_key)
        cv_std = db.fetch_latest_metric_for_strategy(exp_id, sh, split="cv", metric_key=f"{sel_metric}_valid_std")
        cv_ovf = db.fetch_latest_metric_for_strategy(exp_id, sh, split="cv", metric_key=f"{sel_metric}_train_gt_valid_frac")

        # Always show test sharpe for interpretability (if test exists)
        test_sharpe = db.fetch_latest_metric_for_strategy(exp_id, sh, split="test", metric_key="sharpe_net")

        extra_parts: list[str] = []
        if cv_sel is not None:
            extra_parts.append(f"cv_{cv_key}={cv_sel:.4f}")
        if cv_gap is not None and cv_gap == cv_gap:
            extra_parts.append(f"cv_gap={cv_gap:.4f}")
        if cv_std is not None and cv_std == cv_std:
            extra_parts.append(f"cv_std={cv_std:.4f}")
        if cv_ovf is not None and cv_ovf == cv_ovf:
            extra_parts.append(f"cv_train>valid={cv_ovf:.2f}")

        # Selection metric on test: special case for DSR (virtual)
        if sel_metric == "dsr_net":
            sr_per = db.fetch_latest_metric_for_strategy(exp_id, sh, split="test", metric_key="sharpe_net_per")
            n_obs = db.fetch_latest_metric_for_strategy(exp_id, sh, split="test", metric_key="n_obs")
            skew = db.fetch_latest_metric_for_strategy(exp_id, sh, split="test", metric_key="skew_net")
            kurt = db.fetch_latest_metric_for_strategy(exp_id, sh, split="test", metric_key="kurt_net")
            if sr_per is not None and n_obs is not None and skew is not None and kurt is not None:
                dsr_t = deflated_sharpe_ratio(float(sr_per), int(n_obs), float(skew), float(kurt), trials=trials, sr0_per=0.0)
                if dsr_t == dsr_t:  # not NaN
                    extra_parts.append(f"test_dsr_net={dsr_t:.4f}")
        else:
            test_sel = db.fetch_latest_metric_for_strategy(exp_id, sh, split="test", metric_key=sel_metric)
            if test_sel is not None:
                extra_parts.append(f"test_{sel_metric}={test_sel:.4f}")

        if test_sharpe is not None:
            extra_parts.append(f"test_sharpe_net={test_sharpe:.4f}")

        extra = "" if not extra_parts else " | " + " ".join(extra_parts)

        print(
            f"#{row['rank']:02d} {row['score_key']}={row['score_value']:.4f} "
            f"strategy={row['strategy_name']} hash={sh[:12]}{extra}"
        )

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# non-eval change: should NOT break cache
