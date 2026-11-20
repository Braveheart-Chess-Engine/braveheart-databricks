"""The registered rule (`docs/nnue-structural-research.md`): GO, NO-GO or INCONCLUSIVE from
each generation's effects, and from the same effects measured on shuffled features.

A cell is a structure feature in a phase. It is eligible when the sides differ in the
feature in at least MIN_SHARE of the phase's positions in every generation. The
conditions:

- `replication`: at least two generations were analysed.
- `size`: some eligible cell has an effect of at least MIN_EFFECT_CP, with the same sign in
  every generation, each generation's 95% interval excluding zero.
- `control`: on the shuffled features, no cell meets the size condition.
- `power`: every eligible cell would be detected in every generation at once with at least
  POWER_FLOOR, were its effect exactly MIN_EFFECT_CP (and there is an eligible cell).

GO is all four. NO-GO is replication, control and power without size: the data could have
shown an effect of the size that matters and did not. Anything else is INCONCLUSIVE.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .registered import MIN_EFFECT_CP, MIN_SHARE, POWER_FLOOR

_KEY = ["feature", "phase"]


@dataclass(frozen=True)
class Verdict:
    """The outcome, the conditions behind it, and per cell: whether it is `eligible`, its
    `joint_power` (the product of the generations' powers), whether its effect has the
    `same_sign` in every generation, and whether it `meets_size`."""

    outcome: str
    conditions: dict[str, bool]
    cells: pd.DataFrame


def _cells(
    effects: Mapping[int, pd.DataFrame], min_effect_cp: float, min_share: float
) -> pd.DataFrame:
    """The rule's reading of each cell across the generations' effects tables."""
    tables = [table.set_index(_KEY) for table in effects.values()]
    first = tables[0]
    eligible = pd.Series(True, index=first.index)
    positive = pd.Series(True, index=first.index)
    negative = pd.Series(True, index=first.index)
    joint_power = pd.Series(1.0, index=first.index)
    for table in tables:
        table = table.reindex(first.index)
        eligible &= table["share"].fillna(0.0) >= min_share
        # A comparison with a missing effect is false, so a cell without one meets nothing.
        positive &= (table["cp"] >= min_effect_cp) & (table["lo"] > 0)
        negative &= (table["cp"] <= -min_effect_cp) & (table["hi"] < 0)
        joint_power *= table["power"].fillna(0.0)
    signs = pd.concat([(table.reindex(first.index)["cp"] > 0) for table in tables], axis=1)
    known = pd.concat([table.reindex(first.index)["cp"].notna() for table in tables], axis=1)
    same_sign = known.all(axis=1) & (signs.all(axis=1) | ~signs.any(axis=1))
    return pd.DataFrame(
        {
            "eligible": eligible,
            "joint_power": joint_power,
            "same_sign": same_sign,
            "meets_size": eligible & (positive | negative),
        }
    ).reset_index()


def decide(
    effects: Mapping[int, pd.DataFrame],
    shuffled: Mapping[int, pd.DataFrame],
    min_effect_cp: float = MIN_EFFECT_CP,
    min_share: float = MIN_SHARE,
    power_floor: float = POWER_FLOOR,
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
        "power": bool(len(eligible) and (eligible["joint_power"] >= power_floor).all()),
    }
    if all(conditions.values()):
        outcome = "GO"
    elif conditions["replication"] and conditions["control"] and conditions["power"]:
        outcome = "NO-GO"
    else:
        outcome = "INCONCLUSIVE"
    return Verdict(outcome, conditions, cells)
