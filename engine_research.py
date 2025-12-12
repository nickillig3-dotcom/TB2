from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from engine_backtest import run_backtest
from engine_config import EngineConfig
from engine_data import DataSet, make_synthetic_dataset, split_dataset
from engine_metrics import compute_metrics
from engine_persistence import SQLitePersistence
from engine_strategy import build_strategy, generate_candidates
from utils_core import canonical_json, compute_code_hash, set_global_seed


def _load_dataset(cfg: EngineConfig) -> DataSet:
    if cfg.data.type == "synthetic":
        return make_synthetic_dataset(cfg.data, seed=cfg.seed)
    raise ValueError(f"Unknown data.type: {cfg.data.type}")


def run_experiment(cfg: EngineConfig) -> int:
    """Run a research loop defined by config, persist all results, return experiment_id."""
    set_global_seed(cfg.seed)

    # Persistence + experiment snapshot
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

    # Data & splits
    ds = _load_dataset(cfg)
    parts = split_dataset(ds, cfg.splits)

    # Candidate generation
    rng = np.random.default_rng(cfg.seed)
    candidates = list(generate_candidates(cfg.research, cfg.data.n_features, rng))

    # Execute
    per_strategy: Dict[str, Dict[str, Dict[str, float]]] = {}  # strategy_hash -> split -> metrics
    per_strategy_json: Dict[str, str] = {}
    per_strategy_name: Dict[str, str] = {}

    for spec in candidates:
        strategy_hash = spec.spec_hash()
        per_strategy_json[strategy_hash] = canonical_json(spec)
        per_strategy_name[strategy_hash] = spec.name

        strat = build_strategy(spec)
        for split_name, split_ds in parts.items():
            t0 = time.perf_counter()
            run_id = db.start_run(
                experiment_id=exp_id,
                split=split_name,
                strategy_name=spec.name,
                strategy_json=canonical_json(spec),
                strategy_hash=strategy_hash,
            )
            try:
                df = split_ds.frame
                raw_pos = strat.generate_positions(df)
                bt = run_backtest(df, raw_pos, cfg.backtest, cfg.costs)
                metrics = compute_metrics(bt)

                # Light artifacts: equity curve prefix for quick debugging.
                # Full-mode later: store full curves externally (parquet) + reference here.
                artifacts = {
                    "equity_curve_head": {
                        "t0": str(df.index[0]),
                        "t1": str(df.index[-1]),
                        "equity_end": float(bt.equity_curve.iloc[-1]),
                        "equity_head": bt.equity_curve.iloc[: min(50, len(bt.equity_curve))].to_list(),
                    }
                }

                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                db.finish_run(run_id, status="ok", metrics=metrics, elapsed_ms=elapsed_ms, artifacts=artifacts)

                per_strategy.setdefault(strategy_hash, {})[split_name] = metrics
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                db.finish_run(run_id, status="error", metrics={}, elapsed_ms=elapsed_ms, error=repr(e), artifacts=None)

    # Rank strategies by selection split/metric
    sel_split = cfg.research.selection_split
    sel_metric = cfg.research.selection_metric

    ranking: List[dict[str, Any]] = []
    for sh, split_metrics in per_strategy.items():
        m = split_metrics.get(sel_split, {})
        score = float(m.get(sel_metric, float("nan")))
        ranking.append(
            {
                "strategy_hash": sh,
                "strategy_name": per_strategy_name.get(sh, "unknown"),
                "strategy_json": per_strategy_json.get(sh, "{}"),
                "score_value": score,
                "metrics": split_metrics,
            }
        )

    # Drop NaNs and sort
    ranking = [r for r in ranking if np.isfinite(r["score_value"])]
    ranking.sort(key=lambda r: r["score_value"], reverse=True)

    top_k = min(cfg.research.top_k, len(ranking))
    best = ranking[:top_k]

    # Persist winners (experiment-level)
    db.store_best_candidates(exp_id, ranked=best, score_key=sel_metric)

    db.close()
    return exp_id
