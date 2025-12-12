from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math
import logging

from engine_config import RobustnessConfig


logger = logging.getLogger(__name__)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    try:
        num = float(num)
        den = float(den)
    except Exception:
        return float(default)
    if den == 0.0 or math.isnan(num) or math.isnan(den) or math.isinf(num) or math.isinf(den):
        return float(default)
    return float(num / den)


class RobustnessEvaluator:
    """
    Computes a simple robustness score from train/test metrics.

    Output dict contains:
      - robustness_score: 0..1 (higher is better)
      - components: per-component contributions
      - flags: simple overfitting / instability indicators
    """
    def __init__(self, cfg: RobustnessConfig):
        self.cfg = cfg
        self.cfg.validate()

    def evaluate(self, train: Dict[str, float], test: Dict[str, float]) -> Dict[str, Any]:
        tr_sh = float(train.get("sharpe", 0.0) or 0.0)
        te_sh = float(test.get("sharpe", 0.0) or 0.0)

        tr_ret = float(train.get("total_return", 0.0) or 0.0)
        te_ret = float(test.get("total_return", 0.0) or 0.0)

        tr_dd = float(train.get("max_drawdown", 0.0) or 0.0)
        te_dd = float(test.get("max_drawdown", 0.0) or 0.0)

        # Components (each 0..1)
        sharpe_component = self._sharpe_component(tr_sh, te_sh)
        return_component = self._return_component(tr_ret, te_ret)
        dd_component = self._drawdown_component(tr_dd, te_dd)

        base = (
            self.cfg.w_sharpe * sharpe_component
            + self.cfg.w_return * return_component
            + self.cfg.w_drawdown * dd_component
        )

        penalty = self._overfit_penalty(tr_sh, te_sh)

        score = _clip(base * penalty, 0.0, 1.0)

        flags = {
            "test_sharpe_negative": te_sh < 0,
            "train_test_sharpe_gap": tr_sh - te_sh,
            "possible_overfit": (tr_sh - te_sh) > self.cfg.max_overfit_sharpe_gap,
        }

        return {
            "robustness_score": float(score),
            "components": {
                "sharpe_component": float(sharpe_component),
                "return_component": float(return_component),
                "drawdown_component": float(dd_component),
                "penalty": float(penalty),
                "base": float(base),
            },
            "flags": flags,
        }

    def _sharpe_component(self, train_sh: float, test_sh: float) -> float:
        # If train sharpe is non-positive, still allow some credit if test is positive.
        if train_sh <= 0:
            return _clip(0.5 + 0.5 * _clip(test_sh / self.cfg.min_test_sharpe_for_full_score, 0.0, 1.0), 0.0, 1.0)

        ratio = _clip(_safe_ratio(test_sh, train_sh, default=0.0), 0.0, 1.0)
        # Also cap by absolute test performance, so train can't "game" it by being tiny.
        abs_term = _clip(test_sh / self.cfg.min_test_sharpe_for_full_score, 0.0, 1.0)
        return _clip(0.5 * ratio + 0.5 * abs_term, 0.0, 1.0)

    def _return_component(self, train_ret: float, test_ret: float) -> float:
        # Prefer positive test returns; avoid huge penalty if train return is tiny.
        if train_ret <= 0:
            return _clip(0.5 + 0.5 * _clip(test_ret / 0.05, 0.0, 1.0), 0.0, 1.0)
        ratio = _clip(_safe_ratio(test_ret, train_ret, default=0.0), 0.0, 1.0)
        abs_term = _clip(test_ret / 0.10, 0.0, 1.0)
        return _clip(0.5 * ratio + 0.5 * abs_term, 0.0, 1.0)

    def _drawdown_component(self, train_dd: float, test_dd: float) -> float:
        # Lower drawdown is better. If test drawdown <= train drawdown => full score.
        if test_dd <= 0:
            return 1.0
        if train_dd <= 0:
            # No drawdown on train is suspicious; judge only by absolute test drawdown.
            return _clip(1.0 - test_dd, 0.0, 1.0)

        # Ratio: if test drawdown is 2x worse -> 0.5, etc.
        ratio = _safe_ratio(train_dd, test_dd, default=0.0)
        return _clip(ratio, 0.0, 1.0)

    def _overfit_penalty(self, train_sh: float, test_sh: float) -> float:
        gap = train_sh - test_sh
        if gap <= 0:
            return 1.0
        # Soft penalty: gap > max_overfit_sharpe_gap => 0.5, then decays slowly
        if gap <= self.cfg.max_overfit_sharpe_gap:
            return 1.0
        excess = gap - self.cfg.max_overfit_sharpe_gap
        return float(1.0 / (1.0 + 0.5 * excess))
