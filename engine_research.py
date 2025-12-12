from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from engine_backtest import run_backtest
from engine_config import EngineConfig
from engine_cv import CvPlan, build_cv_plan
from engine_data import DataSet, make_synthetic_dataset
from engine_metrics import compute_metrics
from engine_persistence import SQLitePersistence
from engine_strategy import StrategySpec, build_strategy, generate_candidates
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
    """Run a research loop defined by config, persist all results, return experiment_id."""
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

    rng = np.random.default_rng(cfg.seed)
    candidates = list(generate_candidates(cfg.research, cfg.data.n_features, rng))

    sel_metric = cfg.research.selection_metric

    # Cache spec per hash for later holdout evaluation
    spec_by_hash: Dict[str, StrategySpec] = {}

    ranking: List[dict[str, Any]] = []

    for spec in candidates:
        strategy_hash = spec.spec_hash()
        spec_by_hash[strategy_hash] = spec

        valid_vals: List[float] = []
        train_vals: List[float] = []
        fold_metrics_compact: List[Dict[str, Any]] = []

        for fold in plan.folds:
            # Per-fold instance: future-proof for fit() state
            strat = build_strategy(spec).fit(fold.train.frame)

            train_metric_val = float("nan")

            # 1) Train slice (diagnostics)
            t0 = time.perf_counter()
            run_id = db.start_run(
                experiment_id=exp_id,
                split="train",
                fold=fold.fold,
                strategy_name=spec.name,
                strategy_json=canonical_json(spec),
                strategy_hash=strategy_hash,
            )
            try:
                raw_pos_train = strat.generate_positions(fold.train.frame)
                bt_train = run_backtest(fold.train.frame, raw_pos_train, cfg.backtest, cfg.costs)
                m_train = compute_metrics(bt_train)
                train_metric_val = float(m_train.get(sel_metric, float("nan")))

                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                db.finish_run(
                    run_id,
                    status="ok",
                    metrics=m_train,
                    elapsed_ms=elapsed_ms,
                    artifacts=None,
                )
                train_vals.append(train_metric_val)
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                db.finish_run(run_id, status="error", metrics={}, elapsed_ms=elapsed_ms, error=repr(e), artifacts=None)

            # 2) Valid slice (selection / OOS within folds)
            t1 = time.perf_counter()
            run_id2 = db.start_run(
                experiment_id=exp_id,
                split="valid",
                fold=fold.fold,
                strategy_name=spec.name,
                strategy_json=canonical_json(spec),
                strategy_hash=strategy_hash,
            )
            try:
                raw_pos_valid = strat.generate_positions(fold.valid.frame)
                bt_valid = run_backtest(fold.valid.frame, raw_pos_valid, cfg.backtest, cfg.costs)
                m_valid = compute_metrics(bt_valid)
                v = float(m_valid.get(sel_metric, float("nan")))

                elapsed_ms = int((time.perf_counter() - t1) * 1000)
                db.finish_run(
                    run_id2,
                    status="ok",
                    metrics=m_valid,
                    elapsed_ms=elapsed_ms,
                    artifacts={
                        "equity_curve_head": {
                            "t0": str(fold.valid.frame.index[0]),
                            "t1": str(fold.valid.frame.index[-1]),
                            "equity_end": float(bt_valid.equity_curve.iloc[-1]),
                            "equity_head": bt_valid.equity_curve.iloc[: min(50, len(bt_valid.equity_curve))].to_list(),
                        }
                    },
                )

                valid_vals.append(v)

                fold_metrics_compact.append(
                    {
                        "fold": fold.fold,
                        "train_n": int(len(fold.train.frame)),
                        "valid_n": int(len(fold.valid.frame)),
                        "valid_metric": v,
                        "train_metric": train_metric_val,
                    }
                )
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t1) * 1000)
                db.finish_run(run_id2, status="error", metrics={}, elapsed_ms=elapsed_ms, error=repr(e), artifacts=None)

        # Aggregate CV score
        total_folds = len(plan.folds)
        valid_vals_clean = [x for x in valid_vals if np.isfinite(x)]
        n_valid_ok = len(valid_vals_clean)
        complete = (n_valid_ok == total_folds)

        cv_score = float("nan")
        valid_agg = float("nan")
        valid_std = float("nan")
        valid_min = float("nan")

        if complete and n_valid_ok > 0:
            valid_arr = np.asarray(valid_vals_clean, dtype=float)
            valid_agg = _agg(valid_vals_clean, cfg.research.cv_agg)
            valid_std = float(np.std(valid_arr, ddof=1)) if valid_arr.size > 1 else 0.0
            valid_min = float(np.min(valid_arr))
            cv_score = float(valid_agg - float(cfg.research.cv_penalty_std) * valid_std)

        # store CV summary as its own run row (split='cv')
        t2 = time.perf_counter()
        cv_run_id = db.start_run(
            experiment_id=exp_id,
            split="cv",
            fold=None,
            strategy_name=spec.name,
            strategy_json=canonical_json(spec),
            strategy_hash=strategy_hash,
        )
        elapsed_ms = int((time.perf_counter() - t2) * 1000)

        status = "ok" if np.isfinite(cv_score) else "error"
        error = None if status == "ok" else f"incomplete_valid_folds ok={n_valid_ok} total={total_folds}"

        db.finish_run(
            cv_run_id,
            status=status,
            error=error,
            metrics={
                "cv_score": float(cv_score) if np.isfinite(cv_score) else float("nan"),
                "cv_complete": float(1.0 if complete else 0.0),
                "cv_n_folds_ok": float(n_valid_ok),
                "cv_n_folds_total": float(total_folds),
                f"{sel_metric}_valid_{cfg.research.cv_agg}": float(valid_agg),
                f"{sel_metric}_valid_std": float(valid_std),
                f"{sel_metric}_valid_min": float(valid_min),
            },
            elapsed_ms=elapsed_ms,
            artifacts={
                "cv_plan_meta": plan.meta,
                "cv_folds_compact": fold_metrics_compact,
            },
        )

        if np.isfinite(cv_score):
            ranking.append(
                {
                    "strategy_hash": strategy_hash,
                    "strategy_name": spec.name,
                    "strategy_json": canonical_json(spec),
                    "score_value": cv_score,
                }
            )

    ranking.sort(key=lambda r: r["score_value"], reverse=True)
    top_k = min(cfg.research.top_k, len(ranking))
    best = ranking[:top_k]
    db.store_best_candidates(exp_id, ranked=best, score_key="cv_score")

    # Final untouched holdout evaluation for top candidates
    if plan.holdout_test is not None and len(best) > 0:
        for row in best:
            sh = row["strategy_hash"]
            spec = spec_by_hash.get(sh)
            if spec is None:
                continue
            strat = build_strategy(spec).fit(plan.holdout_train.frame)

            t3 = time.perf_counter()
            test_run_id = db.start_run(
                experiment_id=exp_id,
                split="test",
                fold=None,
                strategy_name=spec.name,
                strategy_json=canonical_json(spec),
                strategy_hash=sh,
            )
            try:
                raw_pos_test = strat.generate_positions(plan.holdout_test.frame)
                bt_test = run_backtest(plan.holdout_test.frame, raw_pos_test, cfg.backtest, cfg.costs)
                m_test = compute_metrics(bt_test)
                elapsed_ms = int((time.perf_counter() - t3) * 1000)
                db.finish_run(
                    test_run_id,
                    status="ok",
                    metrics=m_test,
                    elapsed_ms=elapsed_ms,
                    artifacts={
                        "equity_curve_head": {
                            "t0": str(plan.holdout_test.frame.index[0]),
                            "t1": str(plan.holdout_test.frame.index[-1]),
                            "equity_end": float(bt_test.equity_curve.iloc[-1]),
                            "equity_head": bt_test.equity_curve.iloc[: min(50, len(bt_test.equity_curve))].to_list(),
                        }
                    },
                )
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t3) * 1000)
                db.finish_run(test_run_id, status="error", metrics={}, elapsed_ms=elapsed_ms, error=repr(e), artifacts=None)

    db.close()
    return exp_id
