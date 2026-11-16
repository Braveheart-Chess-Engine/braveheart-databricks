"""The report's figures: interactive Plotly charts in one dark theme.

Every figure says in its title which sample it shows, and each generation keeps one colour
everywhere: Plasma, from deep violet (generation 0) to gold (generation 6).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale

from .calibration import SCALE
from .phase import PHASES

LATEST = 6
_BACKGROUND = "#0b0f19"
_GRID = "#232a3b"
_TEXT = "#e6e9ef"


def generation_colour(gen: int) -> str:
    """The colour of a generation in every chart."""
    position = 0.12 + 0.88 * min(max(gen, 0), LATEST) / LATEST
    return sample_colorscale("Plasma", [position])[0]


def _alpha(colour: str, alpha: float) -> str:
    return colour.replace("rgb(", "rgba(").replace(")", f", {alpha})")


THEME = go.layout.Template(pio.templates["plotly_dark"])
THEME.layout.update(
    colorway=[generation_colour(g) for g in range(LATEST + 1)],
    paper_bgcolor=_BACKGROUND,
    plot_bgcolor=_BACKGROUND,
    font={
        "family": "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
        "size": 13,
        "color": _TEXT,
    },
    hoverlabel={"bgcolor": "#161c2b", "bordercolor": "#3a4258", "font": {"size": 12}},
    legend={"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
    xaxis={"gridcolor": _GRID, "zerolinecolor": "#3a4258"},
    yaxis={"gridcolor": _GRID, "zerolinecolor": "#3a4258"},
    margin={"t": 96, "l": 64, "r": 32, "b": 56},
)

_OUTCOME_COLOURS = {"white wins": "#f1f3f5", "draws": "#868e96", "black wins": "#4c6ef5"}
_PHASE_COLOURS = {"opening": "#4dabf7", "middlegame": "#9775fa", "endgame": "#f783ac"}
#: Categories the data names (how games ended), in order: never a generation's colour.
_CATEGORY_PALETTE = [
    "#4dabf7",
    "#adb5bd",
    "#ff6b6b",
    "#ffd43b",
    "#69db7c",
    "#e599f7",
    "#ffa94d",
    "#3bc9db",
    "#f06595",
    "#a9e34b",
]
_SCORE_AXIS = "score (centipawns, white's point of view)"


def _finish(figure: go.Figure, title: str, sample_label: str, **layout: object) -> go.Figure:
    figure.update_layout(
        template=THEME,
        title={
            "text": f"<b>{title}</b><br><sup>{sample_label}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        **layout,
    )
    return figure


def _generations(frame: pd.DataFrame) -> list[int]:
    return sorted(int(g) for g in frame["gen"].unique())


def overview_tiles(overview: Mapping[str, float], sample_label: str) -> go.Figure:
    """The dataset at a glance: positions and games read, generations, the sample's share."""
    tiles = [
        ("positions read", overview["positions"], {"valueformat": ",.0f"}),
        ("games read", overview["games"], {"valueformat": ",.0f"}),
        ("generations", overview["generations"], {"valueformat": "d"}),
        ("of the dataset", 100 * overview["share"], {"valueformat": ".2f", "suffix": "%"}),
    ]
    figure = go.Figure()
    for column, (label, value, number) in enumerate(tiles):
        figure.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                number={
                    **number,
                    "font": {"size": 44, "color": generation_colour(LATEST - column)},
                },
                title={"text": label, "font": {"size": 14}},
                domain={"row": 0, "column": column},
            )
        )
    return _finish(
        figure, "The dataset at a glance", sample_label, grid={"rows": 1, "columns": 4}, height=240
    )


def calibration_curves(table: pd.DataFrame, sample_label: str) -> go.Figure:
    """Per generation, white's mean result in each score bin, against sigmoid(score / 400)."""
    figure = go.Figure()
    for gen in _generations(table):
        rows = table[table["gen"] == gen]
        colour = generation_colour(gen)
        figure.add_trace(
            go.Scatter(
                x=rows["bin"],
                y=rows["mean_result"],
                customdata=rows["n"],
                name=f"gen {gen}",
                mode="lines+markers",
                line={"color": colour, "width": 2.4, "shape": "spline"},
                marker={"color": colour, "size": 5},
                hovertemplate=f"gen {gen}<br>score %{{x}} cp<br>white scores %{{y:.3f}}"
                "<br>%{customdata:,} positions<extra></extra>",
            )
        )
    xs = sorted(table["bin"].unique()) if len(table) else [-2000, 2000]
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=[1 / (1 + math.exp(-x / SCALE)) for x in xs],
            name="sigmoid(score / 400)",
            mode="lines",
            line={"color": "#ffffff", "width": 2, "dash": "dash"},
            hovertemplate="sigmoid(%{x} / 400) = %{y:.3f}<extra></extra>",
        )
    )
    return _finish(
        figure,
        "Do scores predict results?",
        sample_label,
        xaxis_title=_SCORE_AXIS,
        yaxis={"title": "white's mean result", "range": [0, 1]},
        legend={"orientation": "h", "y": -0.18},
        height=560,
    )


def scale_bars(scales: pd.DataFrame, sample_label: str) -> go.Figure:
    """Per generation, the scale that fits its results best, against the trainer's 400."""
    gens = [int(g) for g in scales["gen"]]
    figure = go.Figure(
        go.Bar(
            x=[f"gen {g}" for g in gens],
            y=scales["scale"],
            customdata=scales["n"],
            marker={"color": [generation_colour(g) for g in gens], "line": {"width": 0}},
            text=[f"{s:.0f}" for s in scales["scale"]],
            textposition="outside",
            hovertemplate="%{x}: best scale %{y:.0f}<br>%{customdata:,} positions<extra></extra>",
        )
    )
    figure.add_hline(y=SCALE, line={"color": "#ffffff", "dash": "dash", "width": 2})
    # In the right margin, level with the line: inside, it would sit on a bar's label.
    figure.add_annotation(
        x=1,
        xref="paper",
        xanchor="left",
        xshift=8,
        y=SCALE,
        yref="y",
        text="the trainer's 400",
        showarrow=False,
        font={"color": "#ffffff"},
    )
    return _finish(
        figure,
        "Is 400 still the right scale?",
        sample_label,
        yaxis_title="best-fitting scale (centipawns)",
        showlegend=False,
        height=460,
        margin={"r": 150},
    )


def phase_heatmap(table: pd.DataFrame, sample_label: str) -> go.Figure:
    """Where the fit breaks: mean result minus expected, by score and phase, one generation
    at a time (the latest first; a menu picks another)."""
    gens = _generations(table)
    limit = float(table["error"].abs().max() or 0.05) if len(table) else 0.05
    figure = go.Figure()
    for gen in gens:
        rows = table[table["gen"] == gen]
        grid = rows.pivot_table(index="phase", columns="bin", values="error")
        grid = grid.reindex([p for p in PHASES if p in grid.index])
        figure.add_trace(
            go.Heatmap(
                z=grid.values,
                x=list(grid.columns),
                y=list(grid.index),
                name=f"gen {gen}",
                zmid=0,
                zmin=-limit,
                zmax=limit,
                colorscale="RdBu",
                colorbar={"title": "result − expected", "tickformat": "+.2f"},
                visible=gen == gens[-1],
                hovertemplate=f"gen {gen}<br>%{{y}}, score %{{x}} cp<br>"
                "white scores %{z:+.3f} vs expected<extra></extra>",
            )
        )
    buttons = [
        {"label": f"gen {gen}", "method": "update", "args": [{"visible": [g == gen for g in gens]}]}
        for gen in gens
    ]
    return _finish(
        figure,
        "Where does the fit break?",
        sample_label,
        xaxis_title=_SCORE_AXIS,
        updatemenus=[
            {"buttons": buttons, "active": len(gens) - 1, "x": 1.0, "xanchor": "right", "y": 1.16}
        ],
        height=420,
    )


def _stacked(
    frame: pd.DataFrame, of: str, colours: Mapping[str, str], title: str, sample_label: str
) -> go.Figure:
    figure = go.Figure()
    named = [v for v in colours if v in set(frame[of])]
    others = sorted(set(frame[of]) - set(colours))
    palette = {v: _CATEGORY_PALETTE[i % len(_CATEGORY_PALETTE)] for i, v in enumerate(others)}
    for value in named + others:
        rows = frame[frame[of] == value]
        figure.add_trace(
            go.Bar(
                x=[f"gen {int(g)}" for g in rows["gen"]],
                y=rows["share"],
                name=str(value),
                marker={"color": colours.get(value, palette.get(value)), "line": {"width": 0}},
                hovertemplate=f"%{{x}}: {value} %{{y:.1%}}<extra></extra>",
            )
        )
    return _finish(
        figure,
        title,
        sample_label,
        barmode="stack",
        yaxis={"tickformat": ".0%", "title": "share"},
        legend={"orientation": "h", "y": -0.18},
        height=440,
    )


def results_mix(mix: pd.DataFrame, sample_label: str) -> go.Figure:
    """Per generation, how the positions' games ended."""
    return _stacked(mix, "outcome", _OUTCOME_COLOURS, "Results, by generation", sample_label)


def phase_mix(phases: pd.DataFrame, sample_label: str) -> go.Figure:
    """Per generation, the share of positions in each phase."""
    return _stacked(phases, "phase", _PHASE_COLOURS, "Opening, middlegame, endgame", sample_label)


def terminations(terms: pd.DataFrame, sample_label: str) -> go.Figure:
    """Per generation, the share of games ending each way."""
    return _stacked(terms, "termination", {}, "How games finished, by generation", sample_label)


def _lines(
    frame: pd.DataFrame, x: str, x_title: str, title: str, sample_label: str, fill: bool
) -> go.Figure:
    figure = go.Figure()
    for gen in _generations(frame):
        rows = frame[frame["gen"] == gen].sort_values(x)
        colour = generation_colour(gen)
        figure.add_trace(
            go.Scatter(
                x=rows[x],
                y=rows["share"],
                name=f"gen {gen}",
                mode="lines",
                line={"color": colour, "width": 2, "shape": "spline"},
                fill="tozeroy" if fill else None,
                fillcolor=_alpha(colour, 0.12) if fill else None,
                hovertemplate=f"gen {gen}<br>%{{x}}: %{{y:.2%}}<extra></extra>",
            )
        )
    return _finish(
        figure,
        title,
        sample_label,
        xaxis_title=x_title,
        yaxis={"tickformat": ".1%", "title": "share"},
        legend={"orientation": "h", "y": -0.18},
        height=460,
    )


def score_ridgeline(hist: pd.DataFrame, sample_label: str) -> go.Figure:
    """Per generation, how the scores are spread."""
    return _lines(hist, "bin", _SCORE_AXIS, "How the scores are spread", sample_label, fill=False)


def game_lengths(lengths: pd.DataFrame, sample_label: str) -> go.Figure:
    """Per generation, how long the games ran after the opening."""
    return _lines(
        lengths,
        "plies_bin",
        "engine plies after the opening",
        "How long games ran",
        sample_label,
        fill=False,
    )
