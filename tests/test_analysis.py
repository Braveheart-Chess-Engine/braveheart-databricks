import math

import pandas as pd
import pytest

from selfplay_analysis.analysis import (
    calibration_table,
    completeness,
    coverage,
    coverage_note,
    fitted_scales,
    game_lengths,
    phase_mix,
    results_mix,
    score_histogram,
    skipped_lines,
    terminations,
)

COLUMNS = ["gen", "ok", "phase", "bin", "n", "wins", "draws", "losses"]


def _bins(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def _sigmoid(x: float, scale: float) -> float:
    return 1 / (1 + math.exp(-x / scale))


def _synthetic(gen: int, scale: float, n: int = 1000) -> list[tuple]:
    """Bins whose results follow sigmoid(score / scale) exactly."""
    rows = []
    for centre in range(-1500, 1501, 50):
        wins = round(n * _sigmoid(centre, scale))
        rows.append((gen, True, "middlegame", centre, n, wins, 0, n - wins))
    return rows


def test_mean_result_counts_draws_as_half() -> None:
    table = calibration_table(_bins([(6, True, "opening", 100, 400, 100, 200, 100)]))
    assert table.loc[0, "mean_result"] == pytest.approx(0.5)
    assert table.loc[0, "expected"] == pytest.approx(_sigmoid(100, 400))


def test_thin_bins_are_left_out() -> None:
    bins = _bins(
        [(6, True, "opening", 0, 150, 50, 50, 50), (6, True, "opening", 50, 250, 100, 50, 100)]
    )
    assert calibration_table(bins)["bin"].tolist() == [50]


def test_scores_outside_the_window_are_left_out() -> None:
    bins = _bins(
        [(6, True, "endgame", 2000, 300, 290, 5, 5), (6, True, "endgame", 2050, 300, 299, 1, 0)]
    )
    assert calibration_table(bins)["bin"].tolist() == [2000]


def test_phases_are_kept_apart_when_asked() -> None:
    bins = _bins(
        [(6, True, "opening", 0, 300, 100, 100, 100), (6, True, "endgame", 0, 300, 0, 300, 0)]
    )
    assert set(calibration_table(bins, by_phase=True)["phase"]) == {"opening", "endgame"}
    assert calibration_table(bins)["n"].tolist() == [600]


def test_malformed_lines_are_counted_not_fatal() -> None:
    bins = _bins([(6, False, None, None, 3, 0, 0, 0), (6, True, "opening", 0, 300, 100, 100, 100)])
    assert skipped_lines(bins) == {6: 3}
    assert calibration_table(bins)["n"].tolist() == [300]


def test_fitted_scales_per_generation() -> None:
    scales = fitted_scales(_bins(_synthetic(0, 300) + _synthetic(6, 450))).set_index("gen")["scale"]
    assert scales[0] == pytest.approx(300, rel=0.02)
    assert scales[6] == pytest.approx(450, rel=0.02)


def test_results_and_phase_mixes_are_shares() -> None:
    bins = _bins(
        [(6, True, "opening", 0, 300, 100, 100, 100), (6, True, "endgame", 100, 100, 50, 25, 25)]
    )
    mix = results_mix(bins).set_index("outcome")["share"]
    assert mix["white wins"] == pytest.approx(150 / 400)
    assert mix.sum() == pytest.approx(1)
    phases = phase_mix(bins).set_index("phase")["share"]
    assert phases["opening"] == pytest.approx(0.75)


def test_score_histogram_is_a_density_within_the_window() -> None:
    bins = _bins(
        [(6, True, "opening", 0, 300, 100, 100, 100), (6, True, "opening", 2050, 100, 100, 0, 0)]
    )
    hist = score_histogram(bins)
    assert hist["bin"].tolist() == [0]
    assert hist["share"].sum() == pytest.approx(1)


def test_game_lengths_and_terminations_are_shares() -> None:
    games = pd.DataFrame(
        [(6, "checkmate", 80, 3), (6, "repetition", 120, 1), (5, "checkmate", 80, 2)],
        columns=["gen", "termination", "plies_bin", "n"],
    )
    ends = terminations(games).set_index(["gen", "termination"])["share"]
    assert ends[(6, "checkmate")] == pytest.approx(0.75)
    lengths = game_lengths(games)
    assert lengths.groupby("gen")["share"].sum().tolist() == pytest.approx([1, 1])


def test_completeness_tells_short_shards_from_unreadable_ones() -> None:
    expected = pd.DataFrame({"file": ["a", "b", "c"], "positions": [25_000, 25_000, 25_000]})
    lines = pd.DataFrame({"gen": [6, 6], "file": ["a", "b"], "lines": [25_000, 24_000]})
    assert completeness(lines, expected) == {
        "shards": 3,
        "short": 1,
        "unreadable": 1,
        "missing": 26_000,
    }


def test_completeness_of_complete_shards() -> None:
    expected = pd.DataFrame({"file": ["a"], "positions": [25_000]})
    lines = pd.DataFrame({"gen": [6], "file": ["a"], "lines": [25_000]})
    assert completeness(lines, expected) == {"shards": 1, "short": 0, "unreadable": 0, "missing": 0}


def test_coverage_counts_what_the_calibration_leaves_out() -> None:
    bins = _bins(
        [
            (6, True, "opening", 0, 300, 100, 100, 100),
            (6, True, "opening", 50, 150, 50, 50, 50),  # a thin bin
            (6, True, "opening", 2050, 100, 100, 0, 0),  # outside the window
            (6, False, None, None, 3, 0, 0, 0),  # not records: not positions
        ]
    )
    assert coverage(bins) == {"positions": 550, "shown": 300, "thin": 150, "outside": 100}


def test_a_bin_thin_in_each_phase_is_thin_by_phase_only() -> None:
    bins = _bins(
        [(6, True, "opening", 0, 150, 50, 50, 50), (6, True, "endgame", 0, 150, 50, 50, 50)]
    )
    assert coverage(bins)["shown"] == 300
    assert coverage(bins, by_phase=True) == {
        "positions": 300,
        "shown": 0,
        "thin": 300,
        "outside": 0,
    }


def test_the_coverage_note_states_every_count() -> None:
    note = coverage_note({"positions": 223_000, "shown": 217_000, "thin": 2_000, "outside": 4_000})
    assert "217,000 positions shown" in note and "200" in note and "2,000 cp" in note
    assert "4,000 outside" in note and "2,000 in thinner bins" in note
