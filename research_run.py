from __future__ import annotations

import argparse

from engine_config import load_config
from engine_persistence import SQLitePersistence
from engine_research import run_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strategy-Miner: research runner (walk-forward CV + holdout).")
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--show-top", type=int, default=10, help="Show top-N after run.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    exp_id = run_experiment(cfg)

    db = SQLitePersistence(cfg.persistence.db_path)
    best = db.fetch_best_candidates(exp_id)

    print(f"experiment_id={exp_id} db={cfg.persistence.db_path}")
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
