"""The registered rule (`docs/nnue-structural-research.md`): GO, NO-GO or INCONCLUSIVE from
each generation's effects, and from the same effects measured on shuffled features.

A cell is a structure feature in a phase. It is eligible when the sides differ in the
feature in at least MIN_SHARE of the phase's positions in every generation. The
conditions:

- `replication`: at least two generations were analysed.
- `size`: some eligible cell has an effect of at least MIN_EFFECT_CP, with the same sign in
  every generation, each generation's 95% interval excluding zero.
- `control`: on the shuffled features, no cell meets the size condition.
- `bounded`: every eligible cell's 95% interval lies inside plus or minus MIN_EFFECT_CP in
  every generation (and there is an eligible cell): an effect of the size that matters is
  ruled out everywhere it could be.

GO is replication, size and control. NO-GO is replication, control and bounded. Anything
else is INCONCLUSIVE: an estimate that reaches the threshold in one generation only, or an
interval too wide to say, decides nothing either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .registered import MIN_EFFECT_CP, MIN_SHARE

_KEY = ["feature", "phase"]


@dataclass(frozen=True)
class Verdict:
    """The outcome, the conditions behind it, and per cell: whether it is `eligible`,
    whether its effect has the `same_sign` in every generation, whether it `meets_size`,
    its `bound` (the interval end furthest from zero in any generation) and whether it is
    `bounded` (that bound is inside the threshold)."""

    outcome: str
    conditions: dict[str, bool]
    cells: pd.DataFrame


def _cells(
    effects: Mapping[int, pd.DataFrame], min_effect_cp: float, min_share: float
) -> pd.DataFrame:
    """The rule's reading of each cell across the generations' effects tables."""
    index = next(iter(effects.values())).set_index(_KEY).index
    tables = [table.set_index(_KEY).reindex(index) for table in effects.values()]
    eligible = pd.Series(True, index=index)
    positive = pd.Series(True, index=index)
    negative = pd.Series(True, index=index)
    for table in tables:
        eligible &= table["share"].fillna(0.0) >= min_share
        # A comparison with a missing effect is false, so a cell without one meets nothing.
        positive &= (table["cp"] >= min_effect_cp) & (table["lo"] > 0)
        negative &= (table["cp"] <= -min_effect_cp) & (table["hi"] < 0)
    signs = pd.concat([table["cp"] > 0 for table in tables], axis=1)
    known = pd.concat([table["cp"].notna() for table in tables], axis=1).all(axis=1)
    ends = pd.concat([table[["lo", "hi"]].abs().max(axis=1) for table in tables], axis=1)
    # A missing interval bounds nothing: its bound is missing, and `bounded` is false.
    bound = ends.max(axis=1).where(known)
    return pd.DataFrame(
        {
            "eligible": eligible,
            "same_sign": known & (signs.all(axis=1) | ~signs.any(axis=1)),
            "meets_size": eligible & (positive | negative),
            "bound": bound,
            "bounded": bound < min_effect_cp,
        }
    ).reset_index()


def decide(
    effects: Mapping[int, pd.DataFrame],
    shuffled: Mapping[int, pd.DataFrame],
    min_effect_cp: float = MIN_EFFECT_CP,
    min_share: float = MIN_SHARE,
) -> Verdict:
    """The verdict from each generation's effects table (structural_model.effects) and the
    tables of the same generations with the structure shuffled."""
    cells = _cells(effects, min_effect_cp, min_share)
    control = _cells(shuffled, min_effect_cp, min_share)
    eligible = cells[cells["eligible"]]
    conditions = {
        "replication": len(effects) >= 2,
        "size": bool(cells["meets_size"].any()),
        "control": not control["meets_size"].any(),
        "bounded": bool(len(eligible) and eligible["bounded"].all()),
    }
    if conditions["replication"] and conditions["size"] and conditions["control"]:
        outcome = "GO"
    elif conditions["replication"] and conditions["control"] and conditions["bounded"]:
        outcome = "NO-GO"
    else:
        outcome = "INCONCLUSIVE"
    return Verdict(outcome, conditions, cells)
