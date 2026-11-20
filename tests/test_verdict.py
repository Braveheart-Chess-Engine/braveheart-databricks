import math

import numpy as np
import pandas as pd

from selfplay_analysis.phase import PHASES
from selfplay_analysis.structure import STRUCTURE
from selfplay_analysis.verdict import decide


def table(cp: float, se: float, share: float = 0.3, **cells: tuple[float, float]) -> pd.DataFrame:
    """An effects table in which every cell has this effect, error and share, but for the
    cells named `<feature>__<phase>`, which have their own (cp, se)."""
    rows = []
    for feature in STRUCTURE:
        for phase in PHASES:
            c, s = cells.get(f"{feature}__{phase}", (cp, se))
            power = 0.5 * (1 + math.erf((15.0 / s - 1.96) / math.sqrt(2)))
            rows.append((feature, phase, c, s, c - 1.96 * s, c + 1.96 * s, share, power))
    return pd.DataFrame(
        rows, columns=["feature", "phase", "cp", "se", "lo", "hi", "share", "power"]
    )


QUIET = {5: table(0.5, 4), 6: table(-0.5, 4)}  # what shuffled features should show


def test_go() -> None:
    v = decide({5: table(20, 4), 6: table(18, 4)}, QUIET)
    assert v.outcome == "GO"
    assert v.conditions == {"replication": True, "size": True, "control": True, "power": True}


def test_no_go_when_powered_and_nothing_is_there() -> None:
    v = decide({5: table(1, 4), 6: table(-2, 4)}, QUIET)
    assert v.outcome == "NO-GO" and v.conditions["power"] and not v.conditions["size"]


def test_inconclusive_when_underpowered() -> None:
    v = decide({5: table(1, 12), 6: table(2, 12)}, QUIET)
    assert v.outcome == "INCONCLUSIVE" and not v.conditions["power"]


def test_opposite_signs_do_not_meet_the_size_condition() -> None:
    v = decide({5: table(20, 4), 6: table(-20, 4)}, QUIET)
    assert not v.conditions["size"] and v.outcome == "NO-GO"
    assert not v.cells["same_sign"].any()


def test_a_sure_effect_below_the_threshold_is_not_big_enough() -> None:
    v = decide({5: table(10, 2), 6: table(11, 2)}, QUIET)
    assert not v.conditions["size"] and v.outcome == "NO-GO"


def test_a_big_effect_whose_interval_reaches_zero_is_not_sure_enough() -> None:
    v = decide({5: table(20, 12), 6: table(20, 12)}, QUIET)
    assert not v.conditions["size"] and v.outcome == "INCONCLUSIVE"


def test_the_threshold_must_be_met_in_every_generation() -> None:
    v = decide({5: table(20, 4), 6: table(12, 4)}, QUIET)
    assert not v.conditions["size"] and v.outcome == "NO-GO"


def test_one_cell_is_enough() -> None:
    quiet = table(0.0, 4)
    v = decide(
        {
            5: table(0.0, 4, passed__endgame=(22.0, 4.0)),
            6: table(0.0, 4, passed__endgame=(17.0, 4.0)),
        },
        {5: quiet, 6: quiet},
    )
    assert v.outcome == "GO"
    met = v.cells[v.cells["meets_size"]]
    assert list(zip(met["feature"], met["phase"], strict=True)) == [("passed", "endgame")]


def test_one_underpowered_eligible_cell_blocks_a_no_go() -> None:
    weak = {"doubled__opening": (0.0, 12.0)}
    v = decide({5: table(1, 4, **weak), 6: table(1, 4, **weak)}, QUIET)
    assert v.outcome == "INCONCLUSIVE" and not v.conditions["power"]
    cell = v.cells[(v.cells["feature"] == "doubled") & (v.cells["phase"] == "opening")].iloc[0]
    assert cell["joint_power"] < 0.8 and cell["eligible"]


def test_a_rare_cell_cannot_trigger_or_block() -> None:
    rare = {5: table(40, 30, share=0.01), 6: table(40, 30, share=0.01)}
    v = decide(rare, QUIET)
    assert not v.cells["eligible"].any() and not v.conditions["size"]
    assert not v.conditions["power"] and v.outcome == "INCONCLUSIVE"  # nothing could be detected


def test_a_cell_must_be_eligible_in_every_generation() -> None:
    v = decide({5: table(20, 4), 6: table(18, 4, share=0.01)}, QUIET)
    assert not v.cells["eligible"].any() and not v.conditions["size"]


def test_a_control_that_finds_an_effect_makes_the_run_inconclusive() -> None:
    found = {5: table(20, 4), 6: table(18, 4)}
    with_effect = decide(found, found)
    assert with_effect.outcome == "INCONCLUSIVE" and not with_effect.conditions["control"]
    without = decide({5: table(1, 4), 6: table(-2, 4)}, found)
    assert without.outcome == "INCONCLUSIVE" and not without.conditions["control"]


def test_one_generation_is_inconclusive() -> None:
    v = decide({6: table(20, 4)}, {6: table(0.0, 4)})
    assert v.outcome == "INCONCLUSIVE" and not v.conditions["replication"]


def test_a_cell_with_no_effect_measured_is_not_eligible() -> None:
    missing = table(1, 4)
    missing.loc[0, ["cp", "se", "lo", "hi", "power"]] = np.nan
    missing.loc[0, "share"] = 0.0
    v = decide({5: missing, 6: table(1, 4)}, QUIET)
    assert not v.cells.loc[0, "eligible"] and v.outcome == "NO-GO"


def test_the_cells_table_has_a_row_per_cell() -> None:
    v = decide({5: table(20, 4), 6: table(18, 4)}, QUIET)
    assert list(v.cells.columns) == [
        "feature",
        "phase",
        "eligible",
        "joint_power",
        "same_sign",
        "meets_size",
    ]
    assert len(v.cells) == len(STRUCTURE) * len(PHASES)
    assert v.cells["joint_power"].between(0.9, 1.0).all()
