"""The rule the structural analysis is judged by was registered before any result: the
research document states the same values the code uses, and holds no result."""

import re
from pathlib import Path

from selfplay_analysis import registered

DOC = (Path(__file__).resolve().parent.parent / "docs" / "nnue-structural-research.md").read_text()


def test_the_document_states_the_registered_values() -> None:
    assert registered.MIN_EFFECT_CP == 15.0
    assert re.search(r"`MIN_EFFECT_CP` = 15 cp", DOC)
    assert "5% of that phase's positions" in DOC
    assert registered.SHARDS_PER_GENERATION == 1200
    assert "| `shards_per_generation` | `1200` |" in DOC
    assert registered.MAX_POSITIONS_PER_GAME == 12
    assert "| `max_positions_per_game` | `12` |" in DOC
    assert registered.GENERATIONS == (5, 6)
    assert "| `generations` | `5,6` |" in DOC


def test_the_rule_is_size_control_and_bounds() -> None:
    rule = DOC[DOC.index("## The rule") : DOC.index("## What the data is")]
    assert re.findall(r"^\d\. \*\*(\w+)\.\*\*", rule, re.M) == ["Size", "Control", "Bounds"]
    assert "| **GO** | Conditions 1 and 2 hold |" in rule
    assert "| **NO-GO** | Conditions 2 and 3 hold |" in rule
    # Neither the predictive gain nor a power calculation decides anything.
    assert "is not a condition" in rule and "80% power" not in rule


def test_the_document_says_what_ran_before_the_rule_was_merged() -> None:
    assert "## What ran before the rule was merged" in DOC


def test_the_document_holds_no_result() -> None:
    assert "Results" not in re.findall(r"^## (.+)$", DOC, re.M)
