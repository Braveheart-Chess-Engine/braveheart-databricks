"""What the charts show, from the bin counts that one Spark pass produces.

A bin-count table has a row per generation, parse outcome (`ok`), phase and 50-centipawn
score bin, with the positions in it (`n`) and how their games ended (`wins`, `draws`,
`losses`, from white's point of view). Lines that are not records have `ok` false and no
phase or bin. Everything here is pandas, so it runs on the driver and in tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import FIT_WINDOW, MIN_BIN, SCALE, fit_scale

OUTCOMES = ("white wins", "draws", "black wins")
_COUNTS = ["n", "wins", "draws", "losses"]


def _ok(bins: pd.DataFrame) -> pd.DataFrame:
    return bins[bins["ok"].astype(bool)]


def _shares(frame: pd.DataFrame, by: str, of: str) -> pd.DataFrame:
    """`frame` (gen, `of`, n) with each row's share of its generation."""
    grouped = frame.groupby(["gen", of], as_index=False)["n"].sum()
    grouped["share"] = grouped["n"] / grouped.groupby(by)["n"].transform("sum")
    return grouped.sort_values(["gen", of]).reset_index(drop=True)


def skipped_lines(bins: pd.DataFrame) -> dict[int, int]:
    """Lines that were not records, per generation (corrupt or truncated shards)."""
    bad = bins[~bins["ok"].astype(bool)]
    return {int(gen): int(n) for gen, n in bad.groupby("gen")["n"].sum().items()}


def calibration_table(bins: pd.DataFrame, by_phase: bool = False) -> pd.DataFrame:
    """Per generation (and phase, when asked) and score bin within the fit window: the
    positions, white's mean result (a draw counts half), what sigmoid(score / 400)
    expects, and the difference. Bins under MIN_BIN positions are left out."""
    ok = _ok(bins)
    ok = ok[ok["bin"].abs() <= FIT_WINDOW]
    keys = ["gen", "phase", "bin"] if by_phase else ["gen", "bin"]
    table = ok.groupby(keys, as_index=False)[_COUNTS].sum()
    table = table[table["n"] >= MIN_BIN].copy()
    table["mean_result"] = (table["wins"] + 0.5 * table["draws"]) / table["n"]
    table["expected"] = 1 / (1 + np.exp(-table["bin"].astype(float) / SCALE))
    table["error"] = table["mean_result"] - table["expected"]
    return (
        table[[*keys, "n", "mean_result", "expected", "error"]]
        .sort_values(keys)
        .reset_index(drop=True)
    )


def fitted_scales(bins: pd.DataFrame) -> pd.DataFrame:
    """Per generation, the scale whose sigmoid best predicts its results, fitted on its
    calibration bins weighted by their positions."""
    rows = []
    for gen, table in calibration_table(bins).groupby("gen"):
        scale = fit_scale(table["bin"].tolist(), table["mean_result"].tolist(), table["n"].tolist())
        rows.append({"gen": int(gen), "scale": scale, "n": int(table["n"].sum())})
    return pd.DataFrame(rows, columns=["gen", "scale", "n"])


def results_mix(bins: pd.DataFrame) -> pd.DataFrame:
    """Per generation, the share of positions from games white won, drew or lost."""
    totals = _ok(bins).groupby("gen", as_index=False)[["wins", "draws", "losses"]].sum()
    long = totals.melt(id_vars="gen", var_name="column", value_name="n")
    long["outcome"] = long["column"].map(
        {"wins": OUTCOMES[0], "draws": OUTCOMES[1], "losses": OUTCOMES[2]}
    )
    return _shares(long[["gen", "outcome", "n"]], "gen", "outcome")


def phase_mix(bins: pd.DataFrame) -> pd.DataFrame:
    """Per generation, the share of positions in each phase."""
    return _shares(_ok(bins)[["gen", "phase", "n"]], "gen", "phase")


def score_histogram(bins: pd.DataFrame, limit: int = FIT_WINDOW) -> pd.DataFrame:
    """Per generation, each score bin's share of the positions within ±limit."""
    ok = _ok(bins)
    return _shares(ok[ok["bin"].abs() <= limit][["gen", "bin", "n"]], "gen", "bin")


def game_lengths(games: pd.DataFrame) -> pd.DataFrame:
    """Per generation, each length bin's share of games (engine plies after the opening)."""
    return _shares(games[["gen", "plies_bin", "n"]], "gen", "plies_bin")


def terminations(games: pd.DataFrame) -> pd.DataFrame:
    """Per generation, the share of games ending each way."""
    return _shares(games[["gen", "termination", "n"]], "gen", "termination")


def completeness(lines: pd.DataFrame, expected: pd.DataFrame) -> dict[str, int]:
    """Lines read per positions file (gen, file, lines) against each shard's manifest
    (file, positions): the shards read, those short of their manifest (a truncated file),
    those with nothing read (an unreadable one), and the positions missing."""
    merged = expected.merge(lines[["file", "lines"]], on="file", how="left")
    merged["lines"] = merged["lines"].fillna(0)
    short = merged[merged["lines"] < merged["positions"]]
    return {
        "shards": len(merged),
        "short": int((short["lines"] > 0).sum()),
        "unreadable": int((short["lines"] == 0).sum()),
        "missing": int((short["positions"] - short["lines"]).sum()),
    }


def coverage(bins: pd.DataFrame, by_phase: bool = False) -> dict[str, int]:
    """What a calibration chart shows and leaves out: the positions read, those in bins of
    at least MIN_BIN within the window (shown), those in thinner bins, and those outside
    the window. Bins are per generation (and phase, when the chart splits by phase)."""
    ok = _ok(bins)
    inside = ok[ok["bin"].abs() <= FIT_WINDOW]
    keys = ["gen", "phase", "bin"] if by_phase else ["gen", "bin"]
    sizes = inside.groupby(keys)["n"].sum()
    return {
        "positions": int(ok["n"].sum()),
        "shown": int(sizes[sizes >= MIN_BIN].sum()),
        "thin": int(sizes[sizes < MIN_BIN].sum()),
        "outside": int(ok["n"].sum() - inside["n"].sum()),
    }


def coverage_note(counts: dict[str, int]) -> str:
    """A chart's second subtitle line: what it shows and what it leaves out."""
    return (
        f"{counts['shown']:,} positions shown, in bins of at least {MIN_BIN} within "
        f"±{FIT_WINDOW:,} cp; {counts['outside']:,} outside ±{FIT_WINDOW:,} cp; "
        f"{counts['thin']:,} in thinner bins"
    )
