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
            rows.append((feature, phase, c, s, c - 1.96 * s, c + 1.96 * s, share))
    return pd.DataFrame(rows, columns=["feature", "phase", "cp", "se", "lo", "hi", "share"])


QUIET = {5: table(0.5, 4), 6: table(-0.5, 4)}  # what shuffled features should show


def test_go() -> None:
    v = decide({5: table(20, 4), 6: table(18, 4)}, QUIET)
    assert v.outcome == "GO"
    assert v.conditions == {"replication": True, "size": True, "control": True, "bounded": False}


def test_no_go_when_every_interval_is_inside_the_threshold() -> None:
    v = decide({5: table(1, 4), 6: table(-2, 4)}, QUIET)  # at most 2 + 7.84 = 9.84 cp
    assert v.outcome == "NO-GO"
    assert v.conditions == {"replication": True, "size": False, "control": True, "bounded": True}


def test_an_interval_that_reaches_the_threshold_is_not_bounded() -> None:
    # 7 + 1.96 * 4 = 14.84 is inside 15; 7.2 + 7.84 = 15.04 is not, in one cell of one generation.
    inside = decide({5: table(7, 4), 6: table(7, 4)}, QUIET)
    assert inside.outcome == "NO-GO" and inside.cells["bounded"].all()
    edge = {"doubled__opening": (7.2, 4.0)}
    outside = decide({5: table(7, 4), 6: table(7, 4, **edge)}, QUIET)
    assert outside.outcome == "INCONCLUSIVE" and not outside.conditions["bounded"]
    cell = outside.cells.set_index(["feature", "phase"]).loc[("doubled", "opening")]
    assert not cell["bounded"] and np.isclose(cell["bound"], 7.2 + 1.96 * 4.0)
    assert outside.cells["bounded"].sum() == len(outside.cells) - 1


def test_the_bound_is_the_widest_interval_end_of_any_generation_either_way() -> None:
    v = decide({5: table(2, 3), 6: table(-6, 4)}, QUIET)
    assert np.allclose(v.cells["bound"], 6 + 1.96 * 4)  # generation 6's lower end, -13.84


def test_a_real_effect_at_the_threshold_is_never_called_no_go() -> None:
    # The estimates of a real 15 cp effect fall either side of 15. Below it in one
    # generation the size condition fails, and the interval still reaches past 15.
    v = decide(
        {5: table(1, 2, passed__endgame=(17.0, 3.0)), 6: table(1, 2, passed__endgame=(13.0, 3.0))},
        QUIET,
    )
    assert not v.conditions["size"] and not v.conditions["bounded"]
    assert v.outcome == "INCONCLUSIVE"


def test_imprecise_estimates_are_inconclusive() -> None:
    v = decide({5: table(1, 12), 6: table(2, 12)}, QUIET)
    assert v.outcome == "INCONCLUSIVE" and not v.conditions["bounded"] and not v.conditions["size"]


def test_opposite_signs_do_not_meet_the_size_condition() -> None:
    v = decide({5: table(20, 4), 6: table(-20, 4)}, QUIET)
    assert not v.conditions["size"] and not v.cells["same_sign"].any()
    assert v.outcome == "INCONCLUSIVE"  # neither replicated nor ruled out


def test_a_sure_effect_below_the_threshold_is_ruled_out() -> None:
    v = decide({5: table(10, 2), 6: table(11, 2)}, QUIET)  # at most 11 + 3.92 = 14.92 cp
    assert not v.conditions["size"] and v.outcome == "NO-GO"


def test_a_big_effect_whose_interval_reaches_zero_is_not_sure_enough() -> None:
    v = decide({5: table(20, 12), 6: table(20, 12)}, QUIET)
    assert not v.conditions["size"] and v.outcome == "INCONCLUSIVE"


def test_the_threshold_must_be_met_in_every_generation() -> None:
    v = decide({5: table(20, 4), 6: table(12, 4)}, QUIET)
    assert not v.conditions["size"] and v.outcome == "INCONCLUSIVE"


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


def test_a_rare_cell_cannot_trigger_or_block() -> None:
    wild = {"passed__opening": (40.0, 30.0)}
    v = decide({5: table(1, 4, **wild), 6: table(1, 4, **wild)}, QUIET)
    assert v.outcome == "INCONCLUSIVE"  # eligible there, and far from bounded
    first, second = table(1, 4, **wild), table(1, 4, **wild)
    for frame in (first, second):
        rare = (frame["feature"] == "passed") & (frame["phase"] == "opening")
        frame.loc[rare, "share"] = 0.01
    v = decide({5: first, 6: second}, QUIET)
    assert v.outcome == "NO-GO" and not v.conditions["size"]
    assert v.cells["eligible"].sum() == len(v.cells) - 1


def test_no_eligible_cell_bounds_nothing() -> None:
    rare = {5: table(1, 2, share=0.01), 6: table(1, 2, share=0.01)}
    v = decide(rare, QUIET)
    assert not v.cells["eligible"].any()
    assert not v.conditions["bounded"] and v.outcome == "INCONCLUSIVE"


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
    assert decide({6: table(20, 4)}, {6: table(0.0, 4)}).outcome == "INCONCLUSIVE"
    v = decide({6: table(1, 2)}, {6: table(0.0, 4)})
    assert v.outcome == "INCONCLUSIVE" and not v.conditions["replication"]


def test_a_cell_with_no_effect_measured_is_not_eligible_and_an_eligible_one_is_not_bounded() -> (
    None
):
    missing = table(1, 4)
    missing.loc[0, ["cp", "se", "lo", "hi"]] = np.nan
    missing.loc[0, "share"] = 0.0
    v = decide({5: missing, 6: table(1, 4)}, QUIET)
    assert not v.cells.loc[0, "eligible"] and v.outcome == "NO-GO"
    unmeasured = table(1, 4)
    unmeasured.loc[0, ["cp", "se", "lo", "hi"]] = np.nan  # eligible by its share, yet no effect
    v = decide({5: unmeasured, 6: table(1, 4)}, QUIET)
    assert v.cells.loc[0, "eligible"] and not v.cells.loc[0, "bounded"]
    assert v.outcome == "INCONCLUSIVE"


def test_the_cells_table_has_a_row_per_cell() -> None:
    v = decide({5: table(20, 4), 6: table(18, 4)}, QUIET)
    assert list(v.cells.columns) == [
        "feature",
        "phase",
        "eligible",
        "same_sign",
        "meets_size",
        "bound",
        "bounded",
    ]
    assert len(v.cells) == len(STRUCTURE) * len(PHASES)
