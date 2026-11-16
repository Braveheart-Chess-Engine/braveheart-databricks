"""Game phase by non-pawn material, measured as braveheart-selfplay's datagen audit does."""

from __future__ import annotations

#: Knight and bishop 3, rook 5, queen 9; pawns and kings count nothing.
VALUES = {"n": 3, "b": 3, "r": 5, "q": 9}
#: Both sides' non-pawn material at the start.
START = 62
#: The audit's thresholds: opening at 50 or more, endgame at 20 or less.
OPENING_AT = 50
ENDGAME_AT = 20
PHASES = ("opening", "middlegame", "endgame")


def non_pawn_material(fen: str) -> int:
    """Both sides' non-pawn material, read from the FEN's board field only."""
    board = fen.split(" ", 1)[0]
    return sum(VALUES.get(square.lower(), 0) for square in board)


def phase_of(npm: int) -> str:
    """The phase of a position with `npm` non-pawn material."""
    if npm >= OPENING_AT:
        return "opening"
    if npm > ENDGAME_AT:
        return "middlegame"
    return "endgame"
