"""One shard's files as a table: a row per line of its positions file, with the record,
its game and ply (from the games file), the net's evaluation and both sides' pawn
structure. Pure Python and pandas, so it runs per shard on a Spark executor
(structural_frames.py) and in tests without Spark.

Nothing here is fatal to a run. A line that is not a record has `ok` false. A shard whose
records do not match its games file has `aligned` false on every row and no game: it is
counted, and left out of the model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .games import assign_games, parse_games, ply_of
from .nnue import Net, evaluate
from .phase import VALUES, phase_of
from .records import parse_record
from .registered import FOLDS, MAX_POSITIONS_PER_GAME
from .structure import FEATURES, structure

_SIDES = tuple(f"{feature}_{side}" for feature in FEATURES for side in "wb")
#: Column and Spark type, in order. Integers are the smallest that hold the column: the
#: model's rows are collected to the driver.
_TYPES = {
    "ok": "BOOLEAN",
    "aligned": "BOOLEAN",
    "game": "INT",
    "ply": "SMALLINT",
    "kept": "BOOLEAN",
    "stm": "TINYINT",
    "score_white": "INT",
    "result": "FLOAT",
    "termination": "STRING",
    "npm": "SMALLINT",
    "npm_diff": "SMALLINT",
    "phase": "STRING",
    "position_hash": "BIGINT",
    "eval_white": "INT",
    **dict.fromkeys(_SIDES, "TINYINT"),
}
COLUMNS = tuple(_TYPES)
SCHEMA = ", ".join(f"{name} {kind}" for name, kind in _TYPES.items())
_PANDAS = {
    "BOOLEAN": "boolean",
    "INT": "Int32",
    "SMALLINT": "Int16",
    "TINYINT": "Int8",
    "BIGINT": "Int64",
    "FLOAT": "Float32",
    "STRING": "object",
}


def fold_of(run: str, task: str, worker: int, folds: int = FOLDS) -> int:
    """The fold of a worker group (a run's task's worker): a hash, so the same worker is
    always in the same fold, whichever shards are sampled."""
    digest = hashlib.sha256(f"{run}/{task}/w{worker:03d}".encode()).hexdigest()
    return int(digest[:8], 16) % folds


def thin(game: Sequence[int], cap: int) -> np.ndarray:
    """Which rows to keep so that no game has more than `cap`, given each row's game with
    a game's rows together: a game of n rows over the cap keeps exactly `cap`, evenly
    spaced (row j when j * cap // n steps up)."""
    games = np.asarray(game)
    kept = np.ones(len(games), dtype=bool)
    if not len(games):
        return kept
    starts = np.flatnonzero(np.r_[True, games[1:] != games[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(games)], strict=True):
        n = end - start
        if n > cap:
            j = np.arange(n)
            kept[start:end] = (j * cap) // n != ((j - 1) * cap) // n
    return kept


def _material(board: str) -> tuple[int, int]:
    """White's and black's non-pawn material (phase.py's values)."""
    white = black = 0
    for char in board:
        value = VALUES.get(char.lower())
        if value is None:
            continue
        if char.isupper():
            white += value
        else:
            black += value
    return white, black


def _is_position(board: str) -> bool:
    """Whether a FEN's board field is a chess position the net can evaluate: eight ranks
    of eight squares, pieces only, one king a side and no more than 32 pieces."""
    ranks = board.split("/")
    if len(ranks) != 8 or board.count("K") != 1 or board.count("k") != 1:
        return False
    pieces = 0
    for rank in ranks:
        squares = 0
        for char in rank:
            if char in "12345678":
                squares += int(char)
            elif char in "pnbrqkPNBRQK":
                squares += 1
                pieces += 1
            else:
                return False
        if squares != 8:
            return False
    return pieces <= 32


def _position_hash(board: str, stm: str) -> int:
    digest = hashlib.sha256(f"{board} {stm}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _record(line: str) -> dict | None:
    """A line's record and everything read from its FEN alone, or None when the line is
    not a record with a whole FEN of a position."""
    try:
        record = parse_record(line)
        board, stm = record.fen.split(" ")[:2]
        if stm not in ("w", "b") or not _is_position(board):
            return None
        white, black = structure(record.fen)
        white_npm, black_npm = _material(board)
        row = {
            "fen": record.fen,
            "abs_ply": ply_of(record.fen),
            "stm": 1 if stm == "w" else -1,
            "score_white": record.score_white,
            "result": record.result,
            "npm": white_npm + black_npm,
            "npm_diff": white_npm - black_npm,
            "phase": phase_of(white_npm + black_npm),
            "position_hash": _position_hash(board, stm),
        }
    except (ValueError, IndexError):
        return None
    for feature, ours, theirs in zip(FEATURES, white, black, strict=True):
        row[f"{feature}_w"] = ours
        row[f"{feature}_b"] = theirs
    return row


def shard_rows(
    positions: str, games: str, net: Net | None, cap: int = MAX_POSITIONS_PER_GAME
) -> pd.DataFrame:
    """A row per line of a positions file (the text, uncompressed), given the shard's
    games file. `COLUMNS` in order: `ok` (the line is a record), `aligned` (the shard's
    records were all assigned to its games), `game` and `ply` within it, `kept` (the row
    survives thinning to `cap` positions a game), `stm` (+1 white to move, -1 black),
    the record's `score_white` and `result`, the game's `termination`, `npm` and
    `npm_diff` (non-pawn material, both sides' and white's minus black's), `phase`,
    `position_hash` (of the board and the side to move), `eval_white` (the net's static
    evaluation, null without a net) and each feature of structure.py for white (`_w`)
    and black (`_b`)."""
    parsed = [_record(line) for line in positions.splitlines()]
    records = [row for row in parsed if row is not None]
    assigned: list[tuple[int, int]] | None
    try:
        shard_games = parse_games(games)
        assigned = assign_games(
            [row["abs_ply"] for row in records],
            [row["score_white"] for row in records],
            [row["result"] for row in records],
            shard_games,
        )
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        # Not JSON, not the games file's shape, or not this shard's games.
        assigned = None
    evals = evaluate(net, [row["fen"] for row in records]).tolist() if net is not None else None
    kept = (
        thin([game for game, _ in assigned], cap).tolist()
        if assigned is not None
        else [False] * len(records)
    )
    for index, row in enumerate(records):
        if assigned is not None:
            row["game"], row["ply"] = assigned[index]
            row["termination"] = shard_games[row["game"]].termination
        row["kept"] = kept[index]
        if evals is not None:
            row["eval_white"] = evals[index]
    frame = pd.DataFrame(
        [row if row is not None else {} for row in parsed], columns=list(COLUMNS), dtype=object
    )
    frame["ok"] = [row is not None for row in parsed]
    frame["aligned"] = assigned is not None
    frame["kept"] = [row is not None and row["kept"] for row in parsed]
    return frame.astype({name: _PANDAS[kind] for name, kind in _TYPES.items()})
