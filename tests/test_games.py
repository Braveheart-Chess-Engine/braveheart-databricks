import json

import pytest

from selfplay_analysis.games import Game, assign_games, parse_games, ply_of

G = [
    Game(10, 1.0, "win-adjudication", (5, 6, 7, 8, 9, 10)),
    Game(10, 0.5, "repetition", (1, 2)),
    Game(12, 0.5, "fifty-moves", (3, 4, 5, 6, 7)),
]


def test_ply_of() -> None:
    assert ply_of("8/8/8/8/8/8/8/K1k5 w - - 1 6") == 10
    assert ply_of("8/8/8/8/8/8/8/K1k5 b - - 0 6") == 11


def test_scores_become_whites() -> None:
    line = json.dumps(
        {
            "start_fen": "8/8/8/8/8/8/8/K1k5 w - - 0 6",
            "moves": ["a1a2", "c1c2", "a2a1"],
            "scores": [30, -20, 10],
            "result": 0.5,
            "termination": "repetition",
        }
    )
    assert parse_games(line + "\n") == [Game(10, 0.5, "repetition", (30, 20, 10))]


def test_a_game_that_black_starts_keeps_the_signs_right() -> None:
    line = json.dumps(
        {
            "start_fen": "8/8/8/8/8/8/8/K1k5 b - - 0 6",
            "moves": ["c1c2", "a1a2"],
            "scores": [30, -20],
            "result": 1.0,
            "termination": "checkmate",
        }
    )
    assert parse_games(line) == [Game(11, 1.0, "checkmate", (-30, -20))]


def test_records_follow_their_games_in_order() -> None:
    plies, scores, results = [10, 12, 15, 10, 11], [5, 7, 10, 1, 2], [1.0, 1.0, 1.0, 0.5, 0.5]
    assert assign_games(plies, scores, results, G) == [(0, 0), (0, 2), (0, 5), (1, 0), (1, 1)]


def test_a_game_without_records_is_skipped() -> None:
    plies, scores, results = [10, 12, 15, 12, 14], [5, 7, 10, 3, 5], [1.0, 1.0, 1.0, 0.5, 0.5]
    assert assign_games(plies, scores, results, G) == [(0, 0), (0, 2), (0, 5), (2, 0), (2, 2)]


def test_a_later_ply_with_the_next_games_score_starts_the_next_game() -> None:
    # Both games are drawn and the second's first record is at a later ply than the
    # first's last: only the score tells them apart.
    games = [Game(10, 0.5, "repetition", (1, 2, 3, 4)), Game(10, 0.5, "repetition", (9, 9, 9, 8))]
    assert assign_games([10, 13], [1, 8], [0.5, 0.5], games) == [(0, 0), (1, 3)]


def test_a_score_that_matches_no_game_is_an_error() -> None:
    with pytest.raises(ValueError, match="record 2"):
        assign_games([10, 12], [5, 99], [1.0, 1.0], G)


def test_no_records_and_no_games() -> None:
    assert assign_games([], [], [], []) == []
