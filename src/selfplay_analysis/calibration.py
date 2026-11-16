"""How well a score predicts a result: the trainer's sigmoid, and the scale that fits best.

The trainer's target is `lambda * result + (1 - lambda) * sigmoid(score / 400)`, so a score
is calibrated when, among positions with that score, white scores `sigmoid(score / 400)`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: The scale the trainer divides scores by.
SCALE = 400.0
#: Fits and curves use scores within ±FIT_WINDOW centipawns.
FIT_WINDOW = 2000
#: Bins with fewer positions are left out of curves and fits.
MIN_BIN = 200
#: The scales a fit may return.
SCALE_RANGE = (100.0, 2000.0)


def expected_result(score_white: float, scale: float = SCALE) -> float:
    """White's expected result for a score: the logistic sigmoid of `score / scale`."""
    x = score_white / scale
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softplus(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _log_loss(beta: float, data: list[tuple[float, float, float]]) -> float:
    # -(r log p + (1 - r) log(1 - p)) with p = sigmoid(x) is softplus(x) - r x.
    return math.fsum(w * (_softplus(s * beta) - r * s * beta) for s, r, w in data)


def fit_scale(
    scores: Sequence[float],
    results: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """The scale whose sigmoid best predicts `results` from `scores` (least log loss),
    within SCALE_RANGE. Weights count observations: a bin is its centre, its mean result
    and its size."""
    if len(scores) != len(results) or (weights is not None and len(weights) != len(scores)):
        raise ValueError("scores, results and weights must be the same length")
    if not scores:
        raise ValueError("no positions to fit")
    w = weights if weights is not None else [1.0] * len(scores)
    data = list(zip(scores, results, w, strict=True))
    # Golden-section search on 1 / scale, where the log loss is convex.
    lo, hi = 1.0 / SCALE_RANGE[1], 1.0 / SCALE_RANGE[0]
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = _log_loss(a, data), _log_loss(b, data)
    while hi - lo > 1e-9:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = _log_loss(a, data)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = _log_loss(b, data)
    return 1.0 / ((lo + hi) / 2)
