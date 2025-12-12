from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from engine_config import EngineConfig
from engine_cv import CvPlan, build_cv_plan
from engine_data import DataSet, make_synthetic_dataset
from engine_executor import execute_tasks
from engine_persistence import SQLitePersistence
from engine_strategy import StrategySpec, generate_candidates
from engine_tasks import (
    FoldEvalTask,
    HoldoutEvalTask,
    eval_fold_task,
    eval_holdout_task,
    init_eval_context,
)
from utils_core import canonical_json, compute_code_hash, set_global_seed


def _load_dataset(cfg: EngineConfig) -> DataSet:
    if cfg.data.type == "synthetic":
        return make_synthetic_dataset(cfg.data, seed=cfg.seed)
    raise ValueError(f"Unknown data.type: {cfg.data.type}")


def _agg(values: List[float], method: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if method == "median":
        return float(np.median(arr))
    if method == "mean":
        return float(np.mean(arr))
    if method == "min":
        return float(np.min(arr))
    raise ValueError(f"Unknown agg method: {method}")


def run_experiment(cfg: EngineConfig) -> int:
    """Run research loop (CV + holdout) and persist results."""
    set_global_seed(cfg.seed)

    db = SQLitePersistence(cfg.persistence.db_path)
    code_hash = compute_code_hash(".")
    cfg_json = canonical_json(cfg)

    exp_id = db.create_experiment(
        run_name=cfg.run_name,
        mode=cfg.mode,
        config_json=cfg_json,
        config_hash=cfg.config_hash(),
        code_hash=code_hash,
    )

    ds = _load_dataset(cfg)
    plan: CvPlan = build_cv_plan(ds, cfg.splits, cfg.cv)

    # Store experiment-level meta once (avoid duplicating CV plan per strategy)
    meta_run_id = db.start_run(
        experiment_id=exp_id,
        split="meta",
        fold=None,
        strategy_name="__experiment__",
        strategy_json="{}",
        strategy_hash="__experiment__",
    )
    db.finish_run(
        meta_run_id,
        status="ok",
        metrics={
            "n_rows": float(len(ds.frame)),
            "n_folds": float(len(plan.folds)),
            "has_holdout": float(1.0 if plan.holdout_test is not None else 0.0),
        },
        elapsed_ms=0,
        artifacts={"cv_plan_meta": plan.meta},
    )

    # Candidate generation
    rng = np.random.default_rng(cfg.seed)
    candidates = list(generate_candidates(cfg.research, cfg.data.n_features, rng))

    # Build shared evaluation context (fold data) and initialize in main process
    fold_frames: List[Tuple[Any, Any]] = [(f.train.frame, f.valid.frame) for f in plan.folds]
    context = {
        "folds": fold_frames,
        "holdout_train": plan.holdout_train.frame,
        "holdout_test": None if plan.holdout_test is None else plan.holdout_test.frame,
    }
    init_eval_context(context)

    # Prepare fold tasks (strategy x fold)
    fold_tasks: List[FoldEvalTask] = []
    spec_by_hash: Dict[str, StrategySpec] = {}
    json_by_hash: Dict[str, str] = {}

    for spec in candidates:
        sh = spec.spec_hash()
        spec_json = canonical_json(spec)
        spec_by_hash[sh] = spec
        json_by_hash[sh] = spec_json
        for fold in plan.folds:
            fold_tasks.append(
                FoldEvalTask(
                    task_id=f"{sh}:{fold.fold}",
                    fold=fold.fold,
                    strategy_spec=spec,
                    strategy_hash=sh,
                    strategy_json=spec_json,
                    backtest_cfg=cfg.backtest,
                    cost_cfg=cfg.costs,
                )
            )

    fold_results, exec_stats = execute_tasks(
        eval_fold_task,
        fold_tasks,
        cfg.execution,
        process_initializer=init_eval_context,
        process_initargs=(context,),
    )

    # Store execution stats once (verifies parallel mode + fallbacks)
    exec_run_id = db.start_run(
        experiment_id=exp_id,
        split="exec",
        fold=None,
        strategy_name="__experiment__",
        strategy_json="{}",
        strategy_hash="__experiment__",
    )
    db.finish_run(
        exec_run_id,
        status="ok",
        metrics={
            "exec_n_tasks": float(exec_stats.n_tasks),
            "exec_n_jobs": float(exec_stats.n_jobs),
            "exec_had_fallback": float(1.0 if exec_stats.had_fallback else 0.0),
        },
        elapsed_ms=0,
        artifacts={
            "exec_stats": {
                "mode": exec_stats.mode,
                "n_jobs": exec_stats.n_jobs,
                "n_tasks": exec_stats.n_tasks,
                "had_fallback": exec_stats.had_fallback,
                "fallback_reason": exec_stats.fallback_reason,
            }
        },
    )

    # Persist fold runs + accumulate per-strategy fold metrics
    sel_metric = cfg.research.selection_metric
    n_folds = len(plan.folds)

    per_strategy_valid_vals: Dict[str, List[float]] = {sh: [] for sh in spec_by_hash.keys()}
    per_strategy_fold_compact: Dict[str, List[Dict[str, Any]]] = {sh: [] for sh in spec_by_hash.keys()}

    for res in fold_results:
        sh = res["strategy_hash"]
        fold_id = int(res["fold"])
        spec_name = res["strategy_name"]
        spec_json = res["strategy_json"]

        # Train persist
        train = res["train"]
        run_id_train = db.start_run(
            experiment_id=exp_id,
            split="train",
            fold=fold_id,
            strategy_name=spec_name,
            strategy_json=spec_json,
            strategy_hash=sh,
        )
        db.finish_run(
            run_id_train,
            status=str(train.get("status", "error")),
            error=train.get("error"),
            metrics=train.get("metrics", {}),
            elapsed_ms=int(train.get("elapsed_ms", 0)),
            artifacts=None,
        )

        # Valid persist
        valid = res["valid"]
        run_id_valid = db.start_run(
            experiment_id=exp_id,
            split="valid",
            fold=fold_id,
            strategy_name=spec_name,
            strategy_json=spec_json,
            strategy_hash=sh,
        )
        artifacts = None
        if valid.get("status") == "ok" and valid.get("artifact") is not None:
            artifacts = {"equity_curve_head": valid["artifact"]}

        db.finish_run(
            run_id_valid,
            status=str(valid.get("status", "error")),
            error=valid.get("error"),
            metrics=valid.get("metrics", {}),
            elapsed_ms=int(valid.get("elapsed_ms", 0)),
            artifacts=artifacts,
        )

        # Aggregate fold compact
        v_metrics = valid.get("metrics", {}) if valid.get("status") == "ok" else {}
        t_metrics = train.get("metrics", {}) if train.get("status") == "ok" else {}

        v_val = float(v_metrics.get(sel_metric, float("nan")))
        t_val = float(t_metrics.get(sel_metric, float("nan")))

        per_strategy_fold_compact[sh].append(
            {
                "fold": fold_id,
                "train_n": int(res.get("train_n", 0)),
                "valid_n": int(res.get("valid_n", 0)),
                "train_metric": t_val,
                "valid_metric": v_val,
                "train_status": str(train.get("status", "error")),
                "valid_status": str(valid.get("status", "error")),
            }
        )

        if valid.get("status") == "ok" and np.isfinite(v_val):
            per_strategy_valid_vals[sh].append(v_val)

    # CV summary per strategy + ranking
    ranking: List[dict[str, Any]] = []
    for sh, spec in spec_by_hash.items():
        vals = per_strategy_valid_vals.get(sh, [])
        complete = (len(vals) == n_folds)

        cv_score = float("nan")
        valid_agg = float("nan")
        valid_std = float("nan")
        valid_min = float("nan")

        if complete and len(vals) > 0:
            arr = np.asarray(vals, dtype=float)
            valid_agg = _agg(vals, cfg.research.cv_agg)
            valid_std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            valid_min = float(np.min(arr))
            cv_score = float(valid_agg - float(cfg.research.cv_penalty_std) * valid_std)

        cv_run_id = db.start_run(
            experiment_id=exp_id,
            split="cv",
            fold=None,
            strategy_name=spec.name,
            strategy_json=json_by_hash[sh],
            strategy_hash=sh,
        )
        db.finish_run(
            cv_run_id,
            status="ok" if np.isfinite(cv_score) else "error",
            error=None if np.isfinite(cv_score) else f"incomplete_valid_folds ok={len(vals)} total={n_folds}",
            metrics={
                "cv_score": float(cv_score) if np.isfinite(cv_score) else float("nan"),
                "cv_complete": float(1.0 if complete else 0.0),
                "cv_n_folds_ok": float(len(vals)),
                "cv_n_folds_total": float(n_folds),
                f"{sel_metric}_valid_{cfg.research.cv_agg}": float(valid_agg),
                f"{sel_metric}_valid_std": float(valid_std),
                f"{sel_metric}_valid_min": float(valid_min),
            },
            elapsed_ms=0,
            artifacts={"cv_folds_compact": per_strategy_fold_compact.get(sh, [])},
        )

        if np.isfinite(cv_score):
            ranking.append(
                {
                    "strategy_hash": sh,
                    "strategy_name": spec.name,
                    "strategy_json": json_by_hash[sh],
                    "score_value": cv_score,
                }
            )

    ranking.sort(key=lambda r: r["score_value"], reverse=True)
    top_k = min(cfg.research.top_k, len(ranking))
    best = ranking[:top_k]
    db.store_best_candidates(exp_id, ranked=best, score_key="cv_score")

    # Holdout test for best candidates
    if plan.holdout_test is not None and len(best) > 0:
        holdout_tasks: List[HoldoutEvalTask] = []
        for row in best:
            sh = row["strategy_hash"]
            spec = spec_by_hash.get(sh)
            if spec is None:
                continue
            holdout_tasks.append(
                HoldoutEvalTask(
                    task_id=f"{sh}:holdout",
                    strategy_spec=spec,
                    strategy_hash=sh,
                    strategy_json=json_by_hash[sh],
                    backtest_cfg=cfg.backtest,
                    cost_cfg=cfg.costs,
                )
            )

        holdout_results, _ = execute_tasks(
            eval_holdout_task,
            holdout_tasks,
            cfg.execution,
            process_initializer=init_eval_context,
            process_initargs=(context,),
        )

        for res in holdout_results:
            sh = res["strategy_hash"]
            spec_name = res["strategy_name"]
            spec_json = res["strategy_json"]
            test = res["test"]

            run_id_test = db.start_run(
                experiment_id=exp_id,
                split="test",
                fold=None,
                strategy_name=spec_name,
                strategy_json=spec_json,
                strategy_hash=sh,
            )
            artifacts = None
            if test.get("status") == "ok" and test.get("artifact") is not None:
                artifacts = {"equity_curve_head": test["artifact"]}

            db.finish_run(
                run_id_test,
                status=str(test.get("status", "error")),
                error=test.get("error"),
                metrics=test.get("metrics", {}),
                elapsed_ms=int(test.get("elapsed_ms", 0)),
                artifacts=artifacts,
            )

    db.close()
    return exp_id
