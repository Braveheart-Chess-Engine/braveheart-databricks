from selfplay_analysis.structure import FEATURES, PawnStructure, pawn_bitboards, structure

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIXED = "4k3/7p/3p4/3P4/2P1P3/P7/P7/4K3 w - - 0 1"
PROTECTED = "4k3/p7/2P5/1P6/8/8/8/4K3 w - - 0 1"


def flip(fen: str) -> str:
    """The colour-flipped mirror: ranks reversed, colours and the side to move swapped."""
    board, stm, *rest = fen.split(" ")
    mirrored = "/".join(rank.swapcase() for rank in reversed(board.split("/")))
    return " ".join([mirrored, "b" if stm == "w" else "w", *rest])


def test_the_features_are_the_structures_fields() -> None:
    assert PawnStructure._fields == FEATURES


def test_bitboards_put_a1_at_bit_zero() -> None:
    white, black = pawn_bitboards("4k3/7p/8/8/8/8/P7/4K3")
    assert white == 1 << 8  # a2
    assert black == 1 << 55  # h7


def test_isolated_doubled_islands_and_passed() -> None:
    white, black = structure(MIXED)
    assert white == PawnStructure(5, 2, 1, 2, 1, 2, 0, 3)
    assert black == PawnStructure(2, 2, 0, 2, 0, 1, 0, 2)


def test_a_protected_passed_pawn() -> None:
    white, black = structure(PROTECTED)
    assert white == PawnStructure(2, 0, 0, 1, 1, 1, 1, 6)
    assert black == PawnStructure(1, 1, 0, 1, 0, 0, 0, 0)


def test_a_phalanx_is_connected() -> None:
    white, _ = structure("4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1")
    assert white == PawnStructure(2, 0, 0, 1, 2, 2, 0, 4)


def test_no_pawns() -> None:
    assert structure("4k3/8/8/8/8/8/8/4K3 w - - 0 1") == (PawnStructure(*[0] * 8),) * 2


def test_pawns_one_step_from_promotion() -> None:
    white, black = structure("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1")
    assert (white.passed, white.passed_rank, black.passed, black.passed_rank) == (1, 7, 1, 7)


def test_the_start_position() -> None:
    white, black = structure(START)
    assert white == black == PawnStructure(8, 0, 0, 1, 8, 0, 0, 0)


def test_the_mirror_swaps_the_sides() -> None:
    for fen in (MIXED, PROTECTED, START):
        white, black = structure(fen)
        assert structure(flip(fen)) == (black, white)
