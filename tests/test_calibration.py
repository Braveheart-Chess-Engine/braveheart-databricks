import math
import random

import pytest

from selfplay_analysis.calibration import FIT_WINDOW, MIN_BIN, expected_result, fit_scale


def _draws(scale: float, n: int, seed: int) -> tuple[list[float], list[float]]:
    """Results drawn at a known scale: white wins with probability sigmoid(score / scale)."""
    rng = random.Random(seed)
    scores = [rng.uniform(-1500, 1500) for _ in range(n)]
    results = [1.0 if rng.random() < expected_result(s, scale) else 0.0 for s in scores]
    return scores, results


def test_expected_result_is_one_half_at_zero() -> None:
    assert expected_result(0) == 0.5


def test_expected_result_uses_the_trainers_scale() -> None:
    assert expected_result(400) == pytest.approx(1 / (1 + math.exp(-1)))
    assert expected_result(-400) == pytest.approx(1 - expected_result(400))


def test_fit_scale_recovers_a_known_scale() -> None:
    scores, results = _draws(320, 200_000, seed=0)
    assert fit_scale(scores, results) == pytest.approx(320, rel=0.04)


def test_a_fit_on_bins_matches_the_fit_on_positions() -> None:
    scores, results = _draws(320, 100_000, seed=1)
    bins: dict[int, tuple[int, float]] = {}
    for s, r in zip(scores, results, strict=True):
        centre = round(s / 50) * 50
        n, total = bins.get(centre, (0, 0.0))
        bins[centre] = (n + 1, total + r)
    centres = list(bins)
    counts = [bins[c][0] for c in centres]
    means = [bins[c][1] / bins[c][0] for c in centres]
    assert fit_scale(centres, means, counts) == pytest.approx(fit_scale(scores, results), rel=0.02)


def test_the_fit_window_and_thinnest_bin() -> None:
    assert FIT_WINDOW == 2000
    assert MIN_BIN == 200
