"""One training record, `FEN | score | result`, as braveheart-selfplay writes it."""

from __future__ import annotations

from dataclasses import dataclass

SEPARATOR = " | "
RESULTS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class Record:
    """A position, its search score in centipawns and its game's result, both from
    white's point of view (braveheart-selfplay's `score_white` and result)."""

    fen: str
    score_white: int
    result: float


def parse_record(line: str) -> Record:
    """Parse one record; a line that is not one raises ValueError."""
    parts = line.split(SEPARATOR)
    if len(parts) != 3:
        raise ValueError(f"not a `FEN | score | result` record: {line!r}")
    fen, score, result = (part.strip() for part in parts)
    if not fen:
        raise ValueError(f"a record without a FEN: {line!r}")
    try:
        score_white = int(score)
        value = float(result)
    except ValueError as exc:
        raise ValueError(f"a record whose score or result is not a number: {line!r}") from exc
    if value not in RESULTS:
        raise ValueError(f"a result must be 0, 0.5 or 1: {line!r}")
    return Record(fen, score_white, value)
