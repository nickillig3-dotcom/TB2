from __future__ import annotations

import argparse

from engine_config import load_config
from engine_persistence import SQLitePersistence
from engine_research import run_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Strategy-Miner: research runner (v1 stadium skeleton).")
    p.add_argument("--config", required=True, help="Path to JSON config (light/full are config-level modes).")
    p.add_argument("--show-top", type=int, default=10, help="Show top-N after run.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    exp_id = run_experiment(cfg)

    # Print quick leaderboard from persisted best_candidates
    db = SQLitePersistence(cfg.persistence.db_path)
    best = db.fetch_best_candidates(exp_id)
    db.close()

    print(f"experiment_id={exp_id} db={cfg.persistence.db_path}")
    if not best:
        print("No best candidates stored (check run logs).")
        return 0

    print(f"Top-{min(args.show_top, len(best))} (selection: {best[0]['score_key']})")
    for row in best[: args.show_top]:
        print(
            f"#{row['rank']:02d} {row['score_key']}={row['score_value']:.4f} "
            f"strategy={row['strategy_name']} hash={row['strategy_hash'][:12]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
