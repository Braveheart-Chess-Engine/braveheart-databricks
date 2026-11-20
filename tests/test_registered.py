"""The rule the structural analysis is judged by was registered before any result: the
research document states the same values the code uses, and holds no result."""

import re
from pathlib import Path

from selfplay_analysis import registered

DOC = (Path(__file__).resolve().parent.parent / "docs" / "nnue-structural-research.md").read_text()


def test_the_document_states_the_registered_values() -> None:
    assert registered.MIN_EFFECT_CP == 15.0
    assert re.search(r"`MIN_EFFECT_CP` = 15 cp", DOC)
    assert "5% of that phase's positions" in DOC and "80% power" in DOC
    assert registered.SHARDS_PER_GENERATION == 672
    assert "| `shards_per_generation` | `672` |" in DOC
    assert f"| `max_positions_per_game` | `{registered.MAX_POSITIONS_PER_GAME}` |" in DOC


def test_the_rule_is_size_control_and_power_and_the_gain_is_not_a_condition() -> None:
    rule = DOC[DOC.index("## The rule") : DOC.index("## What the data is")]
    assert re.findall(r"^\d\. \*\*(\w+)\.\*\*", rule, re.M) == ["Size", "Control", "Power"]
    assert "| **GO** | All three conditions hold |" in rule
    assert "| **NO-GO** | Conditions 2 and 3 hold, and condition 1 fails |" in rule
    assert "is not a condition" in rule


def test_the_document_holds_no_result() -> None:
    assert "Results" not in re.findall(r"^## (.+)$", DOC, re.M)
