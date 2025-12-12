from __future__ import annotations

from engine_config import StrategyMinerConfig, configure_logging
from engine_data import default_data_loader
from engine_features import build_default_pipeline
from engine_search import StrategyMiner, RandomThresholdStrategyGenerator
from engine_results import ResultStore


def main():
    cfg = StrategyMinerConfig()
    cfg.engine.mode = "light"          # switch to "full" later
    cfg.data.price_source = "synthetic"
    # cfg.data.price_csv_path = "your_prices.csv"   # for CSV mode
    # cfg.data.max_rows = 5000                      # override if needed

    cfg.apply_mode_defaults()
    configure_logging(cfg.engine)

    # Load data
    loader = default_data_loader(cfg.data)
    bundle = loader.load(cfg.engine, cfg.data)
    df = bundle.df

    # Build features
    pipeline = build_default_pipeline(cfg.features)

    # Mine strategies
    miner = StrategyMiner()
    gen = RandomThresholdStrategyGenerator()
    results = miner.mine(df, cfg, pipeline, gen)

    # Persist
    store = ResultStore(cfg.results)
    run_id = store.start_run(cfg, bundle.meta, notes="demo run")
    store.save_results(run_id, results)

    # Print short summary
    for r in results:
        print(r["strategy_name"], "rank_score=", round(r["rank_score"], 4),
              "robustness=", round(r["robustness"]["robustness_score"], 3),
              "test_sharpe=", round(r["test_metrics"].get("sharpe", 0.0), 3),
              "test_return=", round(r["test_metrics"].get("total_return", 0.0), 3))


if __name__ == "__main__":
    main()
