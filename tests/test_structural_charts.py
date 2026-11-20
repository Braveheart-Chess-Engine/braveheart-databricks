import numpy as np
import pandas as pd

from selfplay_analysis import structural_charts as charts
from selfplay_analysis.charts import THEME
from selfplay_analysis.phase import PHASES
from selfplay_analysis.structural_model import difference_shares
from selfplay_analysis.structure import STRUCTURE
from selfplay_analysis.verdict import decide

LABEL = "sample: 672 shards a generation, 8.6M positions in the model"
OVERVIEW = {
    "positions": 33_600_000,
    "games": 370_000,
    "shards": 1344,
    "total_shards": 46_808,
    "model_rows": 8_600_000,
    "unaligned": 0,
    "skipped": 0,
    "short": 0,
    "unreadable": 0,
    "missing": 0,
}


def table(cp: float, se: float, share: float = 0.3) -> pd.DataFrame:
    rows = []
    for feature in STRUCTURE:
        for phase in PHASES:
            rows.append((feature, phase, cp, se, cp - 1.96 * se, cp + 1.96 * se, share))
    return pd.DataFrame(rows, columns=["feature", "phase", "cp", "se", "lo", "hi", "share"])


def _effects() -> dict[int, pd.DataFrame]:
    """Every cell at 3 and -2 cp, tightly measured, but two: passed pawns in the opening
    (rare) and doubled pawns in the endgame (measured too loosely to rule anything out)."""
    out = {5: table(3, 4), 6: table(-2, 4)}
    for frame in out.values():
        rare = (frame["feature"] == "passed") & (frame["phase"] == "opening")
        frame.loc[rare, "share"] = 0.01
        weak = (frame["feature"] == "doubled") & (frame["phase"] == "endgame")
        frame.loc[weak, ["se", "lo", "hi"]] = [12.0, -21.0, 26.0]
    return out


QUIET = {5: table(0.5, 4), 6: table(-0.5, 4)}
VERDICT = decide(_effects(), QUIET)
GAINS = {
    5: {
        "structure": {"mean": 4e-4, "lo": 2e-4, "hi": 6e-4},
        "shuffled": {"mean": 0.0, "lo": -1e-4, "hi": 1e-4},
    },
    6: {
        "structure": {"mean": 3e-4, "lo": 1e-4, "hi": 5e-4},
        "shuffled": {"mean": -1e-5, "lo": -1e-4, "hi": 1e-4},
    },
}
SUMMARY = pd.DataFrame(
    [
        ("feature", gen, phase, feature, value, n)
        for gen in (5, 6)
        for phase in PHASES
        for feature in STRUCTURE
        for value, n in ((-1, 20), (0, 60), (1, 20))
    ],
    columns=["kind", "gen", "phase", "feature", "value", "n"],
)


def _figures() -> list:
    return [
        charts.coverage_tiles(OVERVIEW, LABEL),
        charts.structure_shares(difference_shares(SUMMARY), 0.05, LABEL),
        charts.effect_forest(_effects(), VERDICT.cells, 15.0, LABEL),
        charts.gain_intervals(GAINS, LABEL),
        charts.bound_grid(VERDICT, 15.0, LABEL),
        charts.verdict_banner(VERDICT, 15.0, LABEL),
    ]


def test_titles_name_the_sample_and_every_figure_uses_the_theme() -> None:
    for figure in _figures():
        assert LABEL in figure.layout.title.text
        assert figure.layout.template.layout.paper_bgcolor == THEME.layout.paper_bgcolor


def test_difference_shares_are_the_positions_where_the_sides_differ() -> None:
    shares = difference_shares(SUMMARY)
    assert list(shares.columns) == ["gen", "phase", "feature", "positions", "share"]
    assert len(shares) == 2 * len(PHASES) * len(STRUCTURE)
    assert set(shares["positions"]) == {100} and np.allclose(shares["share"], 0.4)


def test_the_tiles_say_when_shards_were_left_out() -> None:
    clean = charts.coverage_tiles(OVERVIEW, LABEL)
    assert any("every shard" in a.text for a in clean.layout.annotations)
    broken = charts.coverage_tiles({**OVERVIEW, "unaligned": 2, "short": 1, "missing": 40}, LABEL)
    note = " ".join(a.text for a in broken.layout.annotations)
    assert "2 shards" in note and "do not match their games" in note and "40" in note


def test_shares_have_a_series_per_generation_and_mark_the_minimum() -> None:
    figure = charts.structure_shares(difference_shares(SUMMARY), 0.05, LABEL)
    assert len(figure.data) == 2 * len(PHASES)
    assert sorted({t.name for t in figure.data}) == ["generation 5", "generation 6"]
    assert sum(bool(t.showlegend) for t in figure.data) == 2
    assert sorted(shape.x0 for shape in figure.layout.shapes) == [0.05] * len(PHASES)


def test_the_forest_has_a_series_per_generation_in_each_phase() -> None:
    figure = charts.effect_forest(_effects(), VERDICT.cells, 15.0, LABEL)
    assert len(figure.data) == 2 * len(PHASES)
    assert sum(bool(t.showlegend) for t in figure.data) == 2
    for trace in figure.data:
        assert len(trace.x) == len(STRUCTURE) and trace.error_x.array is not None
    # Never colour alone: the generations differ by symbol too.
    assert len({t.marker.symbol for t in figure.data}) == 2
    assert len({t.marker.color for t in figure.data}) == 2


def test_the_forest_marks_the_threshold_either_way_in_every_phase() -> None:
    figure = charts.effect_forest(_effects(), VERDICT.cells, 15.0, LABEL)
    lines = sorted(shape.x0 for shape in figure.layout.shapes)
    assert lines == [-15.0] * len(PHASES) + [15.0] * len(PHASES)


def test_the_forest_fades_cells_that_are_not_eligible() -> None:
    figure = charts.effect_forest(_effects(), VERDICT.cells, 15.0, LABEL)
    opening = next(t for t in figure.data if t.xaxis == "x")
    opacity = dict(zip(STRUCTURE, opening.marker.opacity, strict=True))
    assert opacity["passed"] < 0.5 and opacity["isolated"] == 1.0


def test_effects_that_are_missing_draw_without_error() -> None:
    missing = table(np.nan, np.nan, share=0.0)
    verdict = decide({5: missing, 6: missing}, QUIET)
    for figure in (
        charts.effect_forest({5: missing, 6: missing}, verdict.cells, 15.0, LABEL),
        charts.bound_grid(verdict, 15.0, LABEL),
        charts.verdict_banner(verdict, 15.0, LABEL),
    ):
        assert LABEL in figure.layout.title.text


def test_the_gain_names_its_two_series_and_carries_their_intervals() -> None:
    figure = charts.gain_intervals(GAINS, LABEL)
    assert [t.name for t in figure.data] == ["structure", "shuffled"]
    structure = figure.data[0]
    assert np.allclose(structure.x, [4.0, 3.0])  # in 1e-4 of log loss
    assert np.allclose(structure.error_x.array, [2.0, 2.0])
    assert np.allclose(structure.error_x.arrayminus, [2.0, 2.0])
    assert len({t.marker.symbol for t in figure.data}) == 2


def test_the_bound_grid_labels_each_cell_and_flags_the_loose_and_the_rare() -> None:
    figure = charts.bound_grid(VERDICT, 15.0, LABEL)
    (heat,) = figure.data
    assert np.array(heat.z, dtype=float).shape == (len(STRUCTURE), len(PHASES))
    text = {
        (feature, phase): heat.text[i][j]
        for i, feature in enumerate(STRUCTURE)
        for j, phase in enumerate(PHASES)
    }
    assert text[("passed", "opening")] == "–"  # not eligible
    assert text[("doubled", "endgame")] == "±26.0 ✗"  # reaches past the threshold
    assert text[("isolated", "opening")] == "±10.8"  # 3 + 7.84 in generation 5


def test_the_forest_can_say_that_the_run_decides_nothing() -> None:
    plain = charts.effect_forest(_effects(), VERDICT.cells, 15.0, LABEL)
    marked = charts.effect_forest(
        _effects(), VERDICT.cells, 15.0, LABEL, caveat="the run decides nothing"
    )
    assert "the run decides nothing" not in plain.layout.title.text
    assert "⚠ the run decides nothing" in marked.layout.title.text


def test_the_banner_states_the_outcome_and_each_condition() -> None:
    figure = charts.verdict_banner(VERDICT, 15.0, LABEL)
    text = " ".join(a.text for a in figure.layout.annotations)
    assert "INCONCLUSIVE" in text  # one eligible cell is not bounded, and none meets the size
    for condition in ("replication", "size", "control", "bounded"):
        assert condition in text
    assert text.count("✓") == 2 and text.count("✗") == 2
    assert "registered rule" in figure.layout.title.text
    found = decide({5: table(20, 4), 6: table(18, 4)}, QUIET)
    go_text = " ".join(a.text for a in charts.verdict_banner(found, 15.0, LABEL).layout.annotations)
    assert "GO" in go_text and "NO-GO needs" in go_text and go_text.count("✓") == 3
    ruled_out = decide({5: table(3, 4), 6: table(-2, 4)}, QUIET)
    no_go = " ".join(
        a.text for a in charts.verdict_banner(ruled_out, 15.0, LABEL).layout.annotations
    )
    assert "<b>NO-GO</b>" in no_go


def test_the_banner_of_a_run_with_other_parameters_does_not_claim_the_registered_result() -> None:
    figure = charts.verdict_banner(VERDICT, 10.0, LABEL, as_registered=False)
    assert "registered rule" not in figure.layout.title.text
    assert "not the registered" in figure.layout.title.text
