import pandas as pd
import plotly.graph_objects as go
import pytest

from selfplay_analysis import charts
from selfplay_analysis.charts import THEME

LABEL = "sample: 24 shards a generation, 4.2M positions"


def _table(by_phase: bool = False) -> pd.DataFrame:
    rows = []
    for gen in (0, 6):
        for phase in ("opening", "endgame") if by_phase else (None,):
            for centre in (-100, 0, 100):
                row = {"gen": gen, "bin": centre, "n": 500, "mean_result": 0.5, "expected": 0.5}
                if phase:
                    row["phase"] = phase
                row["error"] = row["mean_result"] - row["expected"]
                rows.append(row)
    return pd.DataFrame(rows)


def _figures() -> list[go.Figure]:
    shares = pd.DataFrame({"gen": [0, 6], "share": [0.5, 0.5]})
    return [
        charts.overview_tiles(
            {"positions": 4_200_000, "games": 44_000, "generations": 7, "share": 0.002}, LABEL
        ),
        charts.calibration_curves(_table(), LABEL),
        charts.scale_bars(
            pd.DataFrame({"gen": [0, 6], "scale": [380.0, 410.0], "n": [9, 9]}), LABEL
        ),
        charts.phase_heatmap(_table(by_phase=True), LABEL),
        charts.results_mix(shares.assign(outcome="draws"), LABEL),
        charts.phase_mix(shares.assign(phase="opening"), LABEL),
        charts.score_ridgeline(shares.assign(bin=0), LABEL),
        charts.game_lengths(shares.assign(plies_bin=80), LABEL),
        charts.terminations(shares.assign(termination="checkmate"), LABEL),
    ]


def test_titles_name_the_sample() -> None:
    for figure in _figures():
        assert LABEL in figure.layout.title.text


def test_every_figure_uses_the_theme() -> None:
    for figure in _figures():
        assert figure.layout.template.layout.colorway == THEME.layout.colorway
        assert figure.layout.template.layout.paper_bgcolor == THEME.layout.paper_bgcolor


def test_calibration_curves_has_a_reference_line() -> None:
    names = [trace.name for trace in charts.calibration_curves(_table(), LABEL).data]
    assert "sigmoid(score / 400)" in names
    assert {"gen 0", "gen 6"} <= set(names)


def test_scale_bars_mark_the_trainers_400() -> None:
    figure = charts.scale_bars(
        pd.DataFrame({"gen": [0, 6], "scale": [380.0, 410.0], "n": [9, 9]}), LABEL
    )
    assert any(shape.y0 == 400 and shape.y1 == 400 for shape in figure.layout.shapes)


def test_phase_heatmap_centres_its_colours_on_zero_and_shows_the_latest_generation() -> None:
    figure = charts.phase_heatmap(_table(by_phase=True), LABEL)
    heatmaps = [trace for trace in figure.data if isinstance(trace, go.Heatmap)]
    assert heatmaps and all(trace.zmid == 0 for trace in heatmaps)
    assert [trace.visible for trace in heatmaps] == [False, True]  # gen 0 hidden, gen 6 shown


@pytest.mark.parametrize("gen", [0, 6])
def test_generations_keep_one_colour_everywhere(gen: int) -> None:
    colours = {
        trace.marker.color if trace.marker.color is not None else trace.line.color
        for figure in (charts.calibration_curves(_table(), LABEL),)
        for trace in figure.data
        if trace.name == f"gen {gen}"
    }
    assert colours == {charts.generation_colour(gen)}
