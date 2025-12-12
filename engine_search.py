from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from abc import ABC, abstractmethod
import logging
import time
import json

import numpy as np
import pandas as pd

from engine_config import StrategyMinerConfig, SearchConfig
from engine_features import FeaturePipeline
from engine_strategy_base import Strategy, FeatureThresholdStrategy
from engine_backtest import Backtester
from engine_robustness import RobustnessEvaluator


logger = logging.getLogger(__name__)


class StrategyGenerator(ABC):
    """
    Interface for generating candidate Strategy objects.

    Design note:
    - In "full stadium" operation you will implement multiple generators
      (rule programs, genetic programming, Bayesian optimization, ML model search, etc.)
    - StrategyMiner stays stable; only the generator changes.
    """
    @abstractmethod
    def generate(
        self,
        rng: np.random.Generator,
        train_features: pd.DataFrame,
        cfg: SearchConfig,
    ) -> Iterable[Strategy]:
        raise NotImplementedError


class RandomThresholdStrategyGenerator(StrategyGenerator):
    """
    Default lightweight generator:
    - pick a random feature column (filtered by prefix)
    - choose a threshold based on quantiles of |feature|
    - choose direction (trend vs mean_reversion)

    This is only meant to validate the engine loop end-to-end.
    """
    def generate(
        self,
        rng: np.random.Generator,
        train_features: pd.DataFrame,
        cfg: SearchConfig,
    ) -> Iterable[Strategy]:
        cols = [c for c in train_features.columns if c.startswith(cfg.feature_prefix)]
        if not cols:
            raise ValueError(
                f"No feature columns found with prefix='{cfg.feature_prefix}'. "
                f"Available columns: {list(train_features.columns)[:10]}..."
            )

        max_n = int(cfg.max_candidates_per_batch or 50)

        for _ in range(max_n):
            col = str(rng.choice(cols))
            s = train_features[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue

            q = float(rng.choice(cfg.threshold_quantiles))
            thr = float(np.nanquantile(np.abs(s.values), q))
            if not np.isfinite(thr) or thr <= 0:
                continue

            direction = str(rng.choice(cfg.direction_choices))
            yield FeatureThresholdStrategy(
                feature_col=col,
                threshold=thr,
                direction=direction,
                allow_short=True,
                min_hold_bars=1,
                neutral_zone=0.0,
            )


def time_split(df: pd.DataFrame, train_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split.
    """
    if df.empty:
        raise ValueError("Cannot split empty dataframe")
    n = len(df)
    cut = int(max(1, min(n - 1, round(n * train_frac))))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


def _align_for_backtest(features: pd.DataFrame, prices: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features and prices on the same timestamps (intersection).
    """
    idx = features.index.intersection(prices.index)
    return features.loc[idx], prices.loc[idx]


def _stable_config_key(strategy: Strategy) -> str:
    """
    Stable dedup key for strategies based on their config.
    """
    return json.dumps(strategy.to_config(), sort_keys=True, default=str)


@dataclass
class CandidateResult:
    """
    One evaluated candidate strategy.
    """
    strategy_name: str
    strategy_config: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    robustness: Dict[str, Any]
    rank_score: float
    elapsed_ms: float


class StrategyMiner:
    """
    Orchestrates the "generate -> backtest -> robustness -> rank" loop.

    In light-mode we run sequentially. The class is structured so that adding parallel
    evaluation later is localized to _evaluate_many().
    """
    def __init__(
        self,
        backtester: Optional[Backtester] = None,
        robustness: Optional[RobustnessEvaluator] = None,
    ):
        self.backtester = backtester or Backtester()
        self.robustness = robustness  # created once config is known

    def mine(
        self,
        data: pd.DataFrame,
        cfg: StrategyMinerConfig,
        feature_pipeline: FeaturePipeline,
        generator: StrategyGenerator,
    ) -> List[Dict[str, Any]]:
        """
        Runs a mining session and returns top results as dictionaries.

        Output schema is stable and designed for persistence:
          {
            "strategy_name": ...,
            "strategy_config": {...},
            "train_metrics": {...},
            "test_metrics": {...},
            "robustness": {...},
            "rank_score": ...,
            "elapsed_ms": ...,
          }
        """
        cfg.validate()
        if self.robustness is None:
            self.robustness = RobustnessEvaluator(cfg.robustness)

        price_col = cfg.backtest.price_col
        if price_col not in data.columns:
            raise KeyError(f"Missing price column '{price_col}' in data")

        # Split raw data first (chronological)
        train_raw, test_raw = time_split(data, cfg.search.train_frac)

        # Fit on train, transform train
        train_feat = feature_pipeline.fit_transform(train_raw)

        # Transform test with enough history context for rolling features
        lb = feature_pipeline.lookback
        history = train_raw.tail(max(0, lb + 2)) if lb > 0 else train_raw.tail(2)
        test_context = pd.concat([history, test_raw], axis=0)
        test_context_feat = feature_pipeline.transform(test_context)
        test_feat = test_context_feat.loc[test_raw.index.intersection(test_context_feat.index)]

        # Prices aligned to features (because dropna may reduce rows)
        train_feat_aligned, train_prices = _align_for_backtest(train_feat, train_raw[price_col])
        test_feat_aligned, test_prices = _align_for_backtest(test_feat, test_raw[price_col])

        if len(train_feat_aligned) < 50 or len(test_feat_aligned) < 20:
            logger.warning(
                "Very small train/test after feature dropna. "
                "Consider lowering FeatureConfig.dropna or using more rows."
            )

        rng = np.random.default_rng(cfg.engine.random_seed)

        budget = int(cfg.search.max_candidates or 0)
        if budget <= 0:
            raise ValueError("SearchConfig.max_candidates must be > 0 after applying mode defaults")

        top_k = int(cfg.search.top_k or 10)

        evaluated: List[CandidateResult] = []
        seen: set[str] = set()
        attempted = 0
        empty_batches = 0

        # Main loop: try to evaluate up to `budget` candidates.
        while attempted < budget:
            remaining = budget - attempted
            per_batch = int(min(int(cfg.search.max_candidates_per_batch or remaining), remaining))

            # Generator may produce fewer strategies than requested; that's ok.
            batch_cfg = SearchConfig(**{**cfg.search.__dict__, "max_candidates_per_batch": per_batch})

            candidates = list(generator.generate(rng, train_feat_aligned, batch_cfg))
            if not candidates:
                empty_batches += 1
                logger.warning("Generator produced no candidates (empty_batches=%s).", empty_batches)
                if empty_batches >= 5:
                    break
                continue

            for strat in candidates:
                if attempted >= budget:
                    break
                attempted += 1

                key = _stable_config_key(strat)
                if key in seen:
                    continue
                seen.add(key)

                t0 = time.time()
                try:
                    res = self._evaluate_one(
                        strat=strat,
                        train_features=train_feat_aligned,
                        test_features=test_feat_aligned,
                        train_prices=train_prices,
                        test_prices=test_prices,
                        cfg=cfg,
                    )
                except Exception as e:
                    # Candidate failed (bad features, too few trades, etc.) -> skip
                    logger.debug("Candidate evaluation failed: %s", e, exc_info=True)
                    continue

                res.elapsed_ms = (time.time() - t0) * 1000.0  # type: ignore[attr-defined]
                evaluated.append(res)

                # Keep list size manageable: retain a multiple of top_k
                if len(evaluated) > top_k * 20:
                    evaluated = sorted(evaluated, key=lambda r: r.rank_score, reverse=True)[: top_k * 20]

            logger.info("Progress: attempted=%s/%s evaluated=%s unique=%s", attempted, budget, len(evaluated), len(seen))

        # Final rank
        evaluated = sorted(evaluated, key=lambda r: r.rank_score, reverse=True)[:top_k]
        return [self._to_dict(r) for r in evaluated]

    def _evaluate_one(
        self,
        strat: Strategy,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_prices: pd.Series,
        test_prices: pd.Series,
        cfg: StrategyMinerConfig,
    ) -> CandidateResult:
        # Generate signals
        sig_train = strat.generate_signals(train_features)
        sig_test = strat.generate_signals(test_features)

        # Backtests
        bt_train = self.backtester.run(train_prices, sig_train, cfg.backtest)
        bt_test = self.backtester.run(test_prices, sig_test, cfg.backtest)

        # Simple pruning (cheap + helps efficiency a lot at scale)
        if bt_train.metrics.get("num_trades", 0.0) < cfg.search.min_train_trades:
            raise ValueError("Too few trades on train")
        if bt_train.metrics.get("sharpe", 0.0) < cfg.search.min_train_sharpe:
            raise ValueError("Train sharpe below threshold")
        if bt_test.metrics.get("sharpe", 0.0) < cfg.search.min_test_sharpe:
            raise ValueError("Test sharpe below threshold")

        rob = self.robustness.evaluate(bt_train.metrics, bt_test.metrics) if self.robustness else {"robustness_score": 0.0}

        # Rank score: prioritize test sharpe + robustness + positive test return
        test_sh = float(bt_test.metrics.get("sharpe", 0.0) or 0.0)
        test_ret = float(bt_test.metrics.get("total_return", 0.0) or 0.0)
        rob_score = float(rob.get("robustness_score", 0.0) or 0.0)

        # Very simple stable scoring function:
        # - robustness is a multiplier (0..1)
        # - sharpe dominates (>=0), with a small offset so 0 Sharpe isn't auto-zero
        # - positive return helps, negative return doesn't help
        rank = rob_score * max(0.0, test_sh + 0.25) * (1.0 + max(0.0, test_ret))

        return CandidateResult(
            strategy_name=strat.name,
            strategy_config=strat.to_config(),
            train_metrics=bt_train.metrics,
            test_metrics=bt_test.metrics,
            robustness=rob,
            rank_score=float(rank),
            elapsed_ms=0.0,
        )

    @staticmethod
    def _to_dict(r: CandidateResult) -> Dict[str, Any]:
        return {
            "strategy_name": r.strategy_name,
            "strategy_config": r.strategy_config,
            "train_metrics": r.train_metrics,
            "test_metrics": r.test_metrics,
            "robustness": r.robustness,
            "rank_score": r.rank_score,
            "elapsed_ms": r.elapsed_ms,
        }
