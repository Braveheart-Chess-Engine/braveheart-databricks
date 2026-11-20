"""The structural analysis report's figures, in the calibration report's theme
(charts.py): a generation keeps its colour, and here its marker shape too, so the two
generations are never told apart by colour alone.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .charts import _BACKGROUND, _CATEGORY_PALETTE, _TEXT, _finish, generation_colour
from .phase import PHASES
from .structure import STRUCTURE
from .verdict import Verdict

_MUTED = "#868e96"
_GOOD, _BAD = "#69db7c", "#ff6b6b"
_SYMBOLS = ("circle", "diamond", "square", "cross")
#: The gain's two series: the real features and the shuffled control.
_SERIES = {"structure": (_CATEGORY_PALETTE[0], "circle"), "shuffled": (_CATEGORY_PALETTE[6], "x")}
_LABELS = {feature: feature.replace("_", " ") for feature in STRUCTURE}
_CONDITIONS = {
    "replication": "at least two generations were analysed",
    "size": "an eligible cell has an effect of {cp:g} cp or more, the same way in every generation",
    "control": "the shuffled features show no such effect",
    "power": "every eligible cell could have shown one",
}
_OUTCOMES = {
    "GO": "the pawn-pair net comparison is worth its cost",
    "NO-GO": "the data could have shown an effect of the size that matters, and did not",
    "INCONCLUSIVE": "nothing is read from this run",
}


def _marker(colour: str, symbol: str, **more: object) -> dict:
    """A mark of 10 px with a ring in the surface colour, so overlapping marks stay apart."""
    return {
        "color": colour,
        "symbol": symbol,
        "size": 10,
        "line": {"color": _BACKGROUND, "width": 2},
        **more,
    }


def _positions(count: int, index: int, of: int) -> list[float]:
    """The rows 0..count-1 of a categorical axis, moved so that `of` series sit side by
    side on each row."""
    offset = (index - (of - 1) / 2) * 0.28
    return [row + offset for row in range(count)]


def _feature_axis(figure: go.Figure) -> None:
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(STRUCTURE))),
        ticktext=[_LABELS[feature] for feature in STRUCTURE],
        range=[len(STRUCTURE) - 0.5, -0.5],
        showgrid=False,
        zeroline=False,
    )
    figure.update_layout(margin={"t": 144})


def coverage_tiles(overview: Mapping[str, int], sample_label: str) -> go.Figure:
    """What was read: positions, games, shards of all there are, and the rows the models
    are fitted on; and whether anything was left out (shards short of their manifests,
    shards that do not match their games, lines that are not records)."""
    tiles = [
        ("positions read", overview["positions"], {"valueformat": ",.0f"}),
        ("games read", overview["games"], {"valueformat": ",.0f"}),
        (
            "shards read",
            overview["shards"],
            {"valueformat": ",.0f", "suffix": f" of {overview['total_shards']:,}"},
        ),
        ("positions in the model", overview["model_rows"], {"valueformat": ",.0f"}),
    ]
    figure = go.Figure()
    for column, (label, value, number) in enumerate(tiles):
        figure.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                number={**number, "font": {"size": 36, "color": _TEXT}},
                title={"text": label, "font": {"size": 14}},
                domain={"row": 0, "column": column},
            )
        )
    problems = []
    incomplete = overview["short"] + overview["unreadable"]
    if incomplete:
        problems.append(
            f"{incomplete:,} shard{'s' if incomplete != 1 else ''} short or unreadable against "
            f"their manifests ({overview['missing']:,} positions missing)"
        )
    if overview["unaligned"]:
        count = overview["unaligned"]
        problems.append(
            f"{count:,} shard{'s' if count != 1 else ''} that do not match their games, left out"
        )
    if overview["skipped"]:
        problems.append(f"{overview['skipped']:,} lines that are not records, skipped")
    if problems:
        note, colour = "⚠ " + "; ".join(problems), _BAD
    else:
        note, colour = "✓ every shard complete, and matched to its games", _GOOD
    figure.add_annotation(
        text=note,
        x=0.5,
        xref="paper",
        y=-0.08,
        yref="paper",
        showarrow=False,
        font={"color": colour, "size": 14},
    )
    return _finish(
        figure, "What the report read", sample_label, grid={"rows": 1, "columns": 4}, height=260
    )


def structure_shares(shares: pd.DataFrame, min_share: float, sample_label: str) -> go.Figure:
    """How often each feature has anything to say: per phase, the share of positions where
    the sides differ in it, a mark per generation (structural_model.difference_shares). A
    feature left of the line in any generation is not eligible in that phase."""
    figure = make_subplots(
        rows=1, cols=len(PHASES), shared_yaxes=True, subplot_titles=PHASES, horizontal_spacing=0.03
    )
    generations = sorted(int(gen) for gen in shares["gen"].unique())
    for column, phase in enumerate(PHASES, start=1):
        for index, gen in enumerate(generations):
            rows = (
                shares[(shares["gen"] == gen) & (shares["phase"] == phase)]
                .set_index("feature")
                .reindex(STRUCTURE)
            )
            figure.add_trace(
                go.Scatter(
                    x=rows["share"],
                    y=_positions(len(STRUCTURE), index, len(generations)),
                    mode="markers",
                    name=f"generation {gen}",
                    legendgroup=str(gen),
                    showlegend=column == 1,
                    marker=_marker(generation_colour(gen), _SYMBOLS[index % len(_SYMBOLS)]),
                    customdata=np.column_stack(
                        [[_LABELS[f] for f in STRUCTURE], rows["positions"].fillna(0)]
                    ),
                    hovertemplate=(
                        f"gen {gen}, {phase}<br>%{{customdata[0]}}: the sides differ in "
                        "%{x:.1%} of %{customdata[1]:,.0f} positions<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_vline(x=min_share, line={"color": _MUTED, "width": 1}, row=1, col=column)
    figure.update_xaxes(tickformat=".0%", rangemode="tozero", matches="x")
    figure.update_xaxes(title_text="positions where the sides differ", row=1, col=2)
    _finish(
        figure,
        "How often each feature differs between the sides",
        sample_label,
        detail=f"the line is {min_share:.0%}: a feature left of it in any generation is not "
        "eligible in that phase",
        legend={"orientation": "h", "y": -0.18},
        height=480,
    )
    _feature_axis(figure)
    return figure


def effect_forest(
    effects: Mapping[int, pd.DataFrame],
    cells: pd.DataFrame,
    min_effect_cp: float,
    sample_label: str,
    title: str = "What the evaluation misses, by structure",
) -> go.Figure:
    """Each cell's effect and its 95% interval, a panel per phase and a mark per
    generation, with a line at the smallest effect that matters either way. Positive: the
    side with more of the feature scores better than the evaluation says. Cells that are
    not eligible (`cells`, the verdict's) are faded."""
    figure = make_subplots(
        rows=1, cols=len(PHASES), shared_yaxes=True, subplot_titles=PHASES, horizontal_spacing=0.03
    )
    eligible = cells.set_index(["feature", "phase"])["eligible"]
    generations = sorted(effects)
    for column, phase in enumerate(PHASES, start=1):
        for index, gen in enumerate(generations):
            table = effects[gen]
            rows = table[table["phase"] == phase].set_index("feature").reindex(STRUCTURE)
            colour = generation_colour(gen)
            faded = [1.0 if eligible.get((feature, phase), False) else 0.3 for feature in STRUCTURE]
            figure.add_trace(
                go.Scatter(
                    x=rows["cp"],
                    y=_positions(len(STRUCTURE), index, len(generations)),
                    mode="markers",
                    name=f"generation {gen}",
                    legendgroup=str(gen),
                    showlegend=column == 1,
                    marker=_marker(colour, _SYMBOLS[index % len(_SYMBOLS)], opacity=faded),
                    error_x={
                        "type": "data",
                        "array": 1.96 * rows["se"],
                        "color": colour,
                        "thickness": 2,
                        "width": 0,
                    },
                    customdata=np.column_stack(
                        [
                            [_LABELS[f] for f in STRUCTURE],
                            rows["lo"],
                            rows["hi"],
                            rows["share"],
                            rows["power"],
                        ]
                    ),
                    hovertemplate=(
                        f"gen {gen}, {phase}<br>%{{customdata[0]}}: %{{x:+.1f}} cp "
                        "(%{customdata[1]:+.1f} to %{customdata[2]:+.1f})<br>the sides differ in "
                        "%{customdata[3]:.1%} of positions; power %{customdata[4]:.0%}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        for edge in (-min_effect_cp, min_effect_cp):
            figure.add_vline(x=edge, line={"color": _MUTED, "width": 1}, row=1, col=column)
    figure.update_xaxes(matches="x", zeroline=True)
    figure.update_xaxes(title_text="effect (centipawns of evaluation)", row=1, col=2)
    _finish(
        figure,
        title,
        sample_label,
        detail=f"right of zero: the net undervalues the feature; lines at ±{min_effect_cp:g} cp, "
        "the smallest effect that matters; faded cells are not eligible",
        legend={"orientation": "h", "y": -0.18},
        height=540,
    )
    _feature_axis(figure)
    return figure


def gain_intervals(
    gains: Mapping[int, Mapping[str, Mapping[str, float]]], sample_label: str
) -> go.Figure:
    """How much adding the structure terms lowers the log loss of held-out predictions,
    per generation, with its 95% interval from resampling games; and the same for the
    shuffled features. Shown for scale: it is not one of the rule's conditions."""
    figure = go.Figure()
    generations = sorted(gains)
    scale = 1e4
    for index, (series, (colour, symbol)) in enumerate(_SERIES.items()):
        found = [gains[gen][series] for gen in generations]
        figure.add_trace(
            go.Scatter(
                x=[g["mean"] * scale for g in found],
                y=_positions(len(generations), index, len(_SERIES)),
                mode="markers",
                name=series,
                marker=_marker(colour, symbol),
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [(g["hi"] - g["mean"]) * scale for g in found],
                    "arrayminus": [(g["mean"] - g["lo"]) * scale for g in found],
                    "color": colour,
                    "thickness": 2,
                    "width": 0,
                },
                hovertemplate=f"{series}: %{{x:.2f}} × 10⁻⁴<extra></extra>",
            )
        )
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(generations))),
        ticktext=[f"gen {gen}" for gen in generations],
        range=[len(generations) - 0.5, -0.5],
        showgrid=False,
        zeroline=False,
    )
    return _finish(
        figure,
        "Do the structure features predict results better?",
        sample_label,
        detail="the drop in log loss on held-out workers, with its 95% interval; shown for "
        "scale, not a condition of the rule",
        xaxis={"title": "gain in log loss a position (× 10⁻⁴)", "zeroline": True},
        legend={"orientation": "h", "y": -0.3},
        height=300 + 40 * len(generations),
    )


def power_grid(verdict: Verdict, power_floor: float, sample_label: str) -> go.Figure:
    """The chance each eligible cell would have been detected in every generation at once,
    were its effect the smallest that matters. A cell under the floor is marked ✗; a cell
    that is not eligible is blank."""
    cells = verdict.cells.set_index(["feature", "phase"])
    z, text = [], []
    for feature in STRUCTURE:
        z_row, text_row = [], []
        for phase in PHASES:
            cell = cells.loc[(feature, phase)]
            if not cell["eligible"]:
                z_row.append(None)
                text_row.append("–")
                continue
            power = float(cell["joint_power"])
            z_row.append(power)
            text_row.append(f"{power:.0%}" + ("" if power >= power_floor else " ✗"))
        z.append(z_row)
        text.append(text_row)
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=list(PHASES),
            y=[_LABELS[feature] for feature in STRUCTURE],
            text=text,
            texttemplate="%{text}",
            zmin=0,
            zmax=1,
            colorscale=[[0, "#16233b"], [1, _CATEGORY_PALETTE[0]]],
            xgap=2,
            ygap=2,
            colorbar={"title": "power", "tickformat": ".0%"},
            hovertemplate="%{y}, %{x}: %{text}<extra></extra>",
        )
    )
    return _finish(
        figure,
        "Could the report have seen an effect of the size that matters?",
        sample_label,
        detail=f"power in every generation at once; the rule asks for {power_floor:.0%} in every "
        "eligible cell (✗: under it; –: not eligible)",
        xaxis={"showgrid": False, "zeroline": False},
        yaxis={"autorange": "reversed", "showgrid": False, "zeroline": False},
        height=460,
    )


def verdict_banner(verdict: Verdict, min_effect_cp: float, sample_label: str) -> go.Figure:
    """The outcome and the conditions behind it, each with a mark and its name: never
    colour alone."""
    figure = go.Figure()
    figure.add_annotation(
        text=f"<b>{verdict.outcome}</b>",
        x=0,
        xref="paper",
        xanchor="left",
        y=1,
        yref="paper",
        yanchor="top",
        showarrow=False,
        font={"size": 40, "color": _TEXT},
    )
    figure.add_annotation(
        text=_OUTCOMES[verdict.outcome],
        x=0,
        xref="paper",
        xanchor="left",
        y=0.68,
        yref="paper",
        yanchor="top",
        showarrow=False,
        font={"size": 15, "color": _TEXT},
    )
    for row, (name, holds) in enumerate(verdict.conditions.items()):
        mark = f'<span style="color:{_GOOD if holds else _BAD}">{"✓" if holds else "✗"}</span>'
        figure.add_annotation(
            text=f"{mark} <b>{name}</b>: {_CONDITIONS[name].format(cp=min_effect_cp)}",
            x=0,
            xref="paper",
            xanchor="left",
            y=0.46 - 0.15 * row,
            yref="paper",
            yanchor="top",
            showarrow=False,
            font={"size": 14, "color": _TEXT},
        )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _finish(figure, "The verdict, by the registered rule", sample_label, height=360)
