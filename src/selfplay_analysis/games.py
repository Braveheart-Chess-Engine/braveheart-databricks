"""Which game each record of a shard belongs to, from the shard's games file.

A positions file has no game identifier, but it is written a whole game at a time, in the
order of the games file, and each record's score is the games file's score at that ply. So
one pass assigns every record: a record belongs to the current game when its result is the
game's, its ply is past the game's previous record and inside the game, and its score is
the game's score at that ply; otherwise it belongs to a later game. No chess logic is
needed: the ply is read from the FEN's counters.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Game:
    """One game of a shard: the ply its first searched position is at, its result (white's
    point of view), how it ended, and the search's score at each ply (white's point of
    view, where the games file has the side to move's)."""

    start_ply: int
    result: float
    termination: str
    scores_white: tuple[int, ...]


def ply_of(fen: str) -> int:
    """A position's ply from the start of chess, by its FEN's move number and side to
    move: 0 for white's first move, 1 for black's."""
    parts = fen.split(" ")
    return 2 * (int(parts[5]) - 1) + (parts[1] == "b")


def parse_games(text: str) -> list[Game]:
    """The games of a games file, one JSON object a line, in order."""
    games = []
    for line in text.splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        start = ply_of(game["start_fen"])
        scores = tuple(
            score if (start + ply) % 2 == 0 else -score for ply, score in enumerate(game["scores"])
        )
        games.append(Game(start, float(game["result"]), game["termination"], scores))
    return games


def assign_games(
    plies: Sequence[int],
    scores_white: Sequence[int],
    results: Sequence[float],
    games: Sequence[Game],
) -> list[tuple[int, int]]:
    """Each record's game (its index in `games`) and ply within that game, given the
    records' plies, scores and results in file order. A record that fits no remaining game
    raises ValueError naming it: the shard and its games file do not belong together."""
    assigned = []
    current, previous = 0, -1
    for number, (ply, score, result) in enumerate(
        zip(plies, scores_white, results, strict=True), start=1
    ):
        while current < len(games):
            game = games[current]
            index = ply - game.start_ply
            if (
                result == game.result
                and ply > previous
                and 0 <= index < len(game.scores_white)
                and game.scores_white[index] == score
            ):
                break
            current, previous = current + 1, -1
        else:
            raise ValueError(f"record {number}: fits no remaining game of the shard")
        assigned.append((current, index))
        previous = ply
    return assigned
