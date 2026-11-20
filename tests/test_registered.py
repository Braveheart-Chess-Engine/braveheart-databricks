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


def test_the_document_holds_no_result() -> None:
    assert "Results" not in re.findall(r"^## (.+)$", DOC, re.M)
