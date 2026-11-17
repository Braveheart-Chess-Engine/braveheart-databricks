import pytest

from selfplay_analysis.phase import non_pawn_material, phase_of

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Black to move: the side-to-move `b` and the castling `KQkq` are not pieces.
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_the_start_position_has_62_non_pawn_material() -> None:
    assert non_pawn_material(START) == 62
    assert non_pawn_material(AFTER_E4) == 62


def test_pawns_and_kings_count_nothing() -> None:
    assert non_pawn_material("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1") == 0


def test_a_rook_each() -> None:
    assert non_pawn_material("4k2r/8/8/8/8/8/8/R3K3 w - - 0 1") == 10


@pytest.mark.parametrize(
    ("npm", "phase"),
    [
        (62, "opening"),
        (50, "opening"),
        (49, "middlegame"),
        (21, "middlegame"),
        (20, "endgame"),
        (0, "endgame"),
    ],
)
def test_phase_follows_the_engine_audit(npm: int, phase: str) -> None:
    assert phase_of(npm) == phase
