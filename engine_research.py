from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from engine_config import EngineConfig
from engine_cv import CvPlan, build_cv_plan
from engine_data import DataSet, make_synthetic_dataset, load_csv_dataset
from engine_evalcode import compute_eval_code_hash
from engine_executor import execute_tasks
from engine_persistence import SQLitePersistence
from engine_signature import evaluation_hash_from_cfg
from engine_strategy import StrategySpec, generate_candidates
from engine_tasks import (
    FoldEvalTask,
    HoldoutEvalTask,
    eval_fold_task,
    eval_holdout_task,
    init_eval_context,
)
from utils_core import canonical_json, compute_code_hash, set_global_seed
from engine_datafingerprint import data_fingerprint_from_cfg_data


def _load_dataset(cfg: EngineConfig) -> DataSet:
    if cfg.data.type == "synthetic":
        return make_synthetic_dataset(cfg.data, seed=cfg.seed)
    if cfg.data.type == "csv":
        return load_csv_dataset(cfg.data)
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
    set_global_seed(cfg.seed)

    db = SQLitePersistence(cfg.persistence.db_path)

    code_hash = compute_code_hash(".")             # audit/repro full hash
    eval_code_hash = compute_eval_code_hash()      # cache stability hash
    cfg_json = canonical_json(cfg)
    config_hash = cfg.config_hash()
    evaluation_hash = evaluation_hash_from_cfg(cfg)
    # Compute once, reuse in experiment row + meta artifact
    data_fp = data_fingerprint_from_cfg_data(cfg.data, seed=cfg.seed)


    exp_id = db.create_experiment(
        run_name=cfg.run_name,
        mode=cfg.mode,
        config_json=cfg_json,
        config_hash=config_hash,
        evaluation_hash=evaluation_hash,
        eval_code_hash=eval_code_hash,
        code_hash=code_hash,
        data_type=str(data_fp.get("type", "unknown")),
        dataset_id=str(data_fp.get("dataset_id", "unknown")),
        dataset_version=str(data_fp.get("dataset_version", "unknown")),
        data_fp_hash=str(data_fp.get("fingerprint_hash", "")),
    )

    ds = _load_dataset(cfg)
    plan: CvPlan = build_cv_plan(ds, cfg.splits, cfg.cv)

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
        artifacts={
            "cv_plan_meta": plan.meta,
            "data_fingerprint": data_fp,
        },
    )

    exec_run_id = db.start_run(
        experiment_id=exp_id,
        split="exec",
        fold=None,
        strategy_name="__experiment__",
        strategy_json="{}",
        strategy_hash="__experiment__",
    )

    rng = np.random.default_rng(cfg.seed)
    raw_candidates = list(generate_candidates(cfg.research, cfg.data.n_features, rng))

    unique_by_hash: Dict[str, StrategySpec] = {}
    for spec in raw_candidates:
        sh = spec.spec_hash()
        if sh not in unique_by_hash:
            unique_by_hash[sh] = spec
    candidates = list(unique_by_hash.values())

    n_candidates_generated = len(raw_candidates)
    n_candidates_unique = len(candidates)

    fold_frames: List[Tuple[Any, Any]] = [(f.train.frame, f.valid.frame) for f in plan.folds]
    context = {
        "folds": fold_frames,
        "holdout_train": plan.holdout_train.frame,
        "holdout_test": None if plan.holdout_test is None else plan.holdout_test.frame,
    }
    init_eval_context(context)

    sel_metric = cfg.research.selection_metric
    n_folds = len(plan.folds)

    fold_tasks: List[FoldEvalTask] = []
    cached_fold_results: List[Dict[str, Any]] = []

    spec_by_hash: Dict[str, StrategySpec] = {}
    json_by_hash: Dict[str, str] = {}

    cache_fold_tasks_skipped = 0

    for spec in candidates:
        sh = spec.spec_hash()
        spec_json = canonical_json(spec)
        spec_by_hash[sh] = spec
        json_by_hash[sh] = spec_json

        for fold in plan.folds:
            fold_id = int(fold.fold)

            train_ref = db.fetch_cached_run_ref(
                evaluation_hash=evaluation_hash,
                eval_code_hash=eval_code_hash,
                strategy_hash=sh,
                split="train",
                fold=fold_id,
                exclude_experiment_id=exp_id,
            )
            valid_ref = db.fetch_cached_run_ref(
                evaluation_hash=evaluation_hash,
                eval_code_hash=eval_code_hash,
                strategy_hash=sh,
                split="valid",
                fold=fold_id,
                exclude_experiment_id=exp_id,
            )

            if train_ref is not None and valid_ref is not None:
                m_train = db.fetch_run_metrics(train_ref["run_id"])
                m_valid = db.fetch_run_metrics(valid_ref["run_id"])
                a_valid = db.fetch_artifact_json_by_run_id(valid_ref["run_id"], "equity_curve_head")

                cached_fold_results.append(
                    {
                        "task_id": f"{sh}:{fold_id}",
                        "fold": fold_id,
                        "strategy_name": spec.name,
                        "strategy_hash": sh,
                        "strategy_json": spec_json,
                        "train_n": int(len(fold.train.frame)),
                        "valid_n": int(len(fold.valid.frame)),
                        "train": {"status": "ok", "error": None, "metrics": m_train, "elapsed_ms": 0, "cached": True, "cache_ref": train_ref},
                        "valid": {"status": "ok", "error": None, "metrics": m_valid, "elapsed_ms": 0, "artifact": a_valid, "cached": True, "cache_ref": valid_ref},
                    }
                )
                cache_fold_tasks_skipped += 1
                continue

            fold_tasks.append(
                FoldEvalTask(
                    task_id=f"{sh}:{fold_id}",
                    fold=fold_id,
                    strategy_spec=spec,
                    strategy_hash=sh,
                    strategy_json=spec_json,
                    backtest_cfg=cfg.backtest,
                    cost_cfg=cfg.costs,
                )
            )

    fold_results_computed, fold_exec_stats = execute_tasks(
        eval_fold_task,
        fold_tasks,
        cfg.execution,
        process_initializer=init_eval_context,
        process_initargs=(context,),
    )

    fold_results: List[Dict[str, Any]] = []
    fold_results.extend(cached_fold_results)
    fold_results.extend(fold_results_computed)

    per_strategy_valid_vals: Dict[str, List[float]] = {sh: [] for sh in spec_by_hash.keys()}
    per_strategy_fold_compact: Dict[str, List[Dict[str, Any]]] = {sh: [] for sh in spec_by_hash.keys()}

    for res in fold_results:
        sh = res["strategy_hash"]
        fold_id = int(res["fold"])
        spec_name = res["strategy_name"]
        spec_json = res["strategy_json"]

        train = res["train"]
        valid = res["valid"]

        run_id_train = db.start_run(exp_id, "train", spec_name, spec_json, sh, fold=fold_id)
        m_train = dict(train.get("metrics", {}) or {})
        train_cached = bool(train.get("cached", False))
        m_train["cached"] = 1.0 if train_cached else 0.0
        train_artifacts: Dict[str, Any] = {}
        if train_cached and isinstance(train.get("cache_ref"), dict):
            train_artifacts["cache_info"] = {"cached": True, **train["cache_ref"]}
        db.finish_run(
            run_id_train,
            status=str(train.get("status", "error")),
            error=train.get("error"),
            metrics=m_train if str(train.get("status")) == "ok" else {},
            elapsed_ms=int(train.get("elapsed_ms", 0)),
            artifacts=train_artifacts or None,
        )

        run_id_valid = db.start_run(exp_id, "valid", spec_name, spec_json, sh, fold=fold_id)
        m_valid = dict(valid.get("metrics", {}) or {})
        valid_cached = bool(valid.get("cached", False))
        m_valid["cached"] = 1.0 if valid_cached else 0.0
        valid_artifacts: Dict[str, Any] = {}
        if valid.get("status") == "ok" and valid.get("artifact") is not None:
            valid_artifacts["equity_curve_head"] = valid["artifact"]
        if valid_cached and isinstance(valid.get("cache_ref"), dict):
            valid_artifacts["cache_info"] = {"cached": True, **valid["cache_ref"]}

        db.finish_run(
            run_id_valid,
            status=str(valid.get("status", "error")),
            error=valid.get("error"),
            metrics=m_valid if str(valid.get("status")) == "ok" else {},
            elapsed_ms=int(valid.get("elapsed_ms", 0)),
            artifacts=valid_artifacts or None,
        )

        v_val = float(m_valid.get(sel_metric, float("nan"))) if valid.get("status") == "ok" else float("nan")
        t_val = float(m_train.get(sel_metric, float("nan"))) if train.get("status") == "ok" else float("nan")

        per_strategy_fold_compact[sh].append(
            {
                "fold": fold_id,
                "train_n": int(res.get("train_n", 0)),
                "valid_n": int(res.get("valid_n", 0)),
                "train_metric": t_val,
                "valid_metric": v_val,
                "train_status": str(train.get("status", "error")),
                "valid_status": str(valid.get("status", "error")),
                "train_cached": bool(train_cached),
                "valid_cached": bool(valid_cached),
            }
        )
        if valid.get("status") == "ok" and np.isfinite(v_val):
            per_strategy_valid_vals[sh].append(v_val)

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

        cv_run_id = db.start_run(exp_id, "cv", spec.name, json_by_hash[sh], sh, fold=None)
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
            ranking.append({"strategy_hash": sh, "strategy_name": spec.name, "strategy_json": json_by_hash[sh], "score_value": cv_score})

    ranking.sort(key=lambda r: r["score_value"], reverse=True)
    top_k = min(cfg.research.top_k, len(ranking))
    best = ranking[:top_k]
    db.store_best_candidates(exp_id, ranked=best, score_key="cv_score")

    holdout_tasks: List[HoldoutEvalTask] = []
    cached_holdout_results: List[Dict[str, Any]] = []
    cache_test_tasks_skipped = 0

    if plan.holdout_test is not None and len(best) > 0:
        for row in best:
            sh = row["strategy_hash"]
            spec = spec_by_hash.get(sh)
            if spec is None:
                continue

            test_ref = db.fetch_cached_run_ref(
                evaluation_hash=evaluation_hash,
                eval_code_hash=eval_code_hash,
                strategy_hash=sh,
                split="test",
                fold=None,
                exclude_experiment_id=exp_id,
            )
            if test_ref is not None:
                m_test = db.fetch_run_metrics(test_ref["run_id"])
                a_test = db.fetch_artifact_json_by_run_id(test_ref["run_id"], "equity_curve_head")
                cached_holdout_results.append(
                    {
                        "task_id": f"{sh}:holdout",
                        "strategy_name": spec.name,
                        "strategy_hash": sh,
                        "strategy_json": json_by_hash[sh],
                        "test": {"status": "ok", "error": None, "metrics": m_test, "elapsed_ms": 0, "artifact": a_test, "cached": True, "cache_ref": test_ref},
                    }
                )
                cache_test_tasks_skipped += 1
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

        holdout_results_computed, _ = execute_tasks(
            eval_holdout_task,
            holdout_tasks,
            cfg.execution,
            process_initializer=init_eval_context,
            process_initargs=(context,),
        )

        holdout_results: List[Dict[str, Any]] = []
        holdout_results.extend(cached_holdout_results)
        holdout_results.extend(holdout_results_computed)

        for res in holdout_results:
            sh = res["strategy_hash"]
            spec_name = res["strategy_name"]
            spec_json = res["strategy_json"]
            test = res["test"]

            run_id_test = db.start_run(exp_id, "test", spec_name, spec_json, sh, fold=None)
            m_test = dict(test.get("metrics", {}) or {})
            test_cached = bool(test.get("cached", False))
            m_test["cached"] = 1.0 if test_cached else 0.0

            test_artifacts: Dict[str, Any] = {}
            if test.get("status") == "ok" and test.get("artifact") is not None:
                test_artifacts["equity_curve_head"] = test["artifact"]
            if test_cached and isinstance(test.get("cache_ref"), dict):
                test_artifacts["cache_info"] = {"cached": True, **test["cache_ref"]}

            db.finish_run(
                run_id_test,
                status=str(test.get("status", "error")),
                error=test.get("error"),
                metrics=m_test if str(test.get("status")) == "ok" else {},
                elapsed_ms=int(test.get("elapsed_ms", 0)),
                artifacts=test_artifacts or None,
            )

    fold_tasks_total = int(n_candidates_unique) * int(n_folds)
    fold_tasks_executed = int(fold_exec_stats.n_tasks)
    fold_tasks_skipped = int(cache_fold_tasks_skipped)

    test_tasks_total = int(top_k) if (plan.holdout_test is not None) else 0
    test_tasks_executed = int(len(holdout_tasks))
    test_tasks_skipped = int(cache_test_tasks_skipped)

    mode_display = fold_exec_stats.mode
    if fold_tasks_total > 0 and fold_tasks_executed == 0 and fold_tasks_skipped == fold_tasks_total:
        mode_display = "cache"

    db.finish_run(
        exec_run_id,
        status="ok",
        metrics={
            "exec_n_tasks": float(fold_exec_stats.n_tasks),
            "exec_n_jobs": float(fold_exec_stats.n_jobs),
            "exec_had_fallback": float(1.0 if fold_exec_stats.had_fallback else 0.0),
            "candidates_generated": float(n_candidates_generated),
            "candidates_unique": float(n_candidates_unique),
            "cache_fold_tasks_total": float(fold_tasks_total),
            "cache_fold_tasks_executed": float(fold_tasks_executed),
            "cache_fold_tasks_skipped": float(fold_tasks_skipped),
            "cache_test_tasks_total": float(test_tasks_total),
            "cache_test_tasks_executed": float(test_tasks_executed),
            "cache_test_tasks_skipped": float(test_tasks_skipped),
        },
        elapsed_ms=0,
        artifacts={
            "exec_stats": {
                "mode": mode_display,
                "requested_n_jobs": int(cfg.execution.n_jobs),
                "prefer_processes": bool(cfg.execution.prefer_processes),
                "n_jobs_used": int(fold_exec_stats.n_jobs),
                "n_tasks_executed": int(fold_exec_stats.n_tasks),
                "had_fallback": bool(fold_exec_stats.had_fallback),
                "fallback_reason": fold_exec_stats.fallback_reason,
                "candidates_generated": int(n_candidates_generated),
                "candidates_unique": int(n_candidates_unique),
                "fold_tasks_total": int(fold_tasks_total),
                "fold_tasks_executed": int(fold_tasks_executed),
                "fold_tasks_skipped": int(fold_tasks_skipped),
                "test_tasks_total": int(test_tasks_total),
                "test_tasks_executed": int(test_tasks_executed),
                "test_tasks_skipped": int(test_tasks_skipped),
                "cache_key": {
                    "evaluation_hash": evaluation_hash,
                    "eval_code_hash": eval_code_hash,
                    "config_hash": config_hash,
                    "code_hash": code_hash,
                },
            }
        },
    )

    db.close()
    return exp_id
