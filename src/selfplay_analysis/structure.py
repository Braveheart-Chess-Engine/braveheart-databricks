"""Pawn-structure features of a position, from the FEN's board field alone.

Each side's pawns are a 64-bit integer, bit `8 * rank + file` with a1 at bit 0. A side is
measured as if it moved up the board: black's pawns are flipped to white's side first, so
both go through the same code and a colour-flipped position gives the sides swapped.

The definitions:

- `pawns`: pawns on the board.
- `isolated`: pawns with no friendly pawn on a neighbouring file.
- `doubled`: pawns beyond the first on a file.
- `islands`: runs of neighbouring files that hold a pawn.
- `connected`: pawns defended by a friendly pawn, or with a friendly pawn beside them on
  the same rank.
- `passed`: pawns with no enemy pawn ahead on their own or a neighbouring file.
- `protected_passed`: passed pawns defended by a friendly pawn.
- `passed_rank`: the most advanced passed pawn's rank counted from its own side (2 to 7),
  0 with none.
"""

from __future__ import annotations

from typing import NamedTuple

FEATURES = (
    "pawns",
    "isolated",
    "doubled",
    "islands",
    "connected",
    "passed",
    "protected_passed",
    "passed_rank",
)


class PawnStructure(NamedTuple):
    """One side's pawn structure (the module documentation defines each field)."""

    pawns: int
    isolated: int
    doubled: int
    islands: int
    connected: int
    passed: int
    protected_passed: int
    passed_rank: int


_FILE_A = 0x0101010101010101
_FILE_H = _FILE_A << 7
_ALL = (1 << 64) - 1
_FILES = tuple(_FILE_A << file for file in range(8))


def _ahead(square: int) -> int:
    """The squares ahead of `square` on its own and the neighbouring files."""
    file, rank = square % 8, square // 8
    span = sum(_FILES[f] for f in (file - 1, file, file + 1) if 0 <= f < 8)
    return span & (_ALL << (8 * (rank + 1))) & _ALL


_AHEAD = tuple(_ahead(square) for square in range(64))


def _flip(pawns: int) -> int:
    """The pawns seen from the other side of the board (ranks reversed)."""
    return int.from_bytes(pawns.to_bytes(8, "little"), "big")


def pawn_bitboards(board: str) -> tuple[int, int]:
    """White's and black's pawns in a FEN's board field."""
    white = black = 0
    for row, rank in enumerate(board.split("/")):
        square = 8 * (7 - row)
        for char in rank:
            if char.isdigit():
                square += int(char)
                continue
            if char == "P":
                white |= 1 << square
            elif char == "p":
                black |= 1 << square
            square += 1
    return white, black


def _side(own: int, enemy: int) -> PawnStructure:
    """The structure of `own`, a side moving up the board, against `enemy`'s pawns."""
    counts = [(own & file).bit_count() for file in _FILES]
    isolated = doubled = islands = 0
    for file, count in enumerate(counts):
        if not count:
            continue
        left = counts[file - 1] if file > 0 else 0
        right = counts[file + 1] if file < 7 else 0
        if not left and not right:
            isolated += count
        if not left:
            islands += 1
        doubled += count - 1
    defended = (((own & ~_FILE_H) << 9) | ((own & ~_FILE_A) << 7)) & _ALL
    beside = ((own & ~_FILE_H) << 1) | ((own & ~_FILE_A) >> 1)
    passed = protected = rank = 0
    remaining = own
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        square = bit.bit_length() - 1
        if enemy & _AHEAD[square]:
            continue
        passed += 1
        protected += bool(bit & defended)
        rank = max(rank, square // 8 + 1)
    return PawnStructure(
        pawns=sum(counts),
        isolated=isolated,
        doubled=doubled,
        islands=islands,
        connected=(own & (defended | beside)).bit_count(),
        passed=passed,
        protected_passed=protected,
        passed_rank=rank,
    )


def structure(fen: str) -> tuple[PawnStructure, PawnStructure]:
    """White's and black's pawn structure in a position."""
    white, black = pawn_bitboards(fen.split(" ", 1)[0])
    return _side(white, black), _side(_flip(black), _flip(white))
