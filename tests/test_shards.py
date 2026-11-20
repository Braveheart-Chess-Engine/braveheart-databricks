import json
import struct
from types import SimpleNamespace

import pytest

from selfplay_analysis import nnue
from selfplay_analysis.registered import FOLDS
from selfplay_analysis.shards import COLUMNS, SCHEMA, fold_of, shard_rows, thin
from selfplay_analysis.structure import FEATURES

MIXED = "4k3/7p/3p4/3P4/2P1P3/P7/P7/4K3"
ROOKS = "r3k3/7p/3p4/3P4/2P1P3/P7/P7/R3K2R"


def _game(start_fen: str, white_scores: list[int], result: float, termination: str) -> str:
    """A games-file line whose scores are given from white's point of view (the file
    holds the side to move's); the game starts with white to move."""
    scores = [score if ply % 2 == 0 else -score for ply, score in enumerate(white_scores)]
    return json.dumps(
        {
            "start_fen": start_fen,
            "moves": ["a1a2"] * len(scores),
            "scores": scores,
            "result": result,
            "termination": termination,
        }
    )


@pytest.fixture
def shard() -> SimpleNamespace:
    """Two games from move 6: three records of a win, then two of a draw."""
    positions = "\n".join(
        [
            f"{MIXED} w - - 0 6 | 50 | 1.0",
            f"{ROOKS} w KQq - 0 7 | 70 | 1.0",
            f"{MIXED} b - - 0 8 | 100 | 1.0",
            f"{MIXED} w - - 0 6 | 5 | 0.5",
            f"{ROOKS} w KQq - 3 7 | 7 | 0.5",
        ]
    )
    games = "\n".join(
        [
            _game(f"{MIXED} w - - 0 6", [50, 60, 70, 80, 90, 100], 1.0, "win-adjudication"),
            _game(f"{MIXED} w - - 0 6", [5, 6, 7, 8], 0.5, "repetition"),
        ]
    )
    return SimpleNamespace(positions=positions + "\n", games=games + "\n")


def test_rows_carry_their_game_ply_and_ending(shard) -> None:
    rows = shard_rows(shard.positions, shard.games, net=None)
    assert rows["game"].tolist() == [0, 0, 0, 1, 1] and rows["ply"].tolist() == [0, 2, 5, 0, 2]
    assert rows["termination"].tolist() == ["win-adjudication"] * 3 + ["repetition"] * 2
    assert rows["ok"].all() and rows["aligned"].all() and rows["eval_white"].isna().all()
    assert rows.loc[0, ["isolated_w", "passed_b", "stm"]].tolist() == [2, 1, 1]
    assert tuple(rows.columns) == COLUMNS


def test_rows_carry_the_record_and_its_material(shard) -> None:
    rows = shard_rows(shard.positions, shard.games, net=None)
    assert rows["score_white"].tolist() == [50, 70, 100, 5, 7]
    assert rows["result"].tolist() == [1.0, 1.0, 1.0, 0.5, 0.5]
    assert rows["stm"].tolist() == [1, 1, -1, 1, 1]
    assert rows["npm"].tolist() == [0, 15, 0, 0, 15]  # ROOKS: white's two rooks, black's one
    assert rows["npm_diff"].tolist() == [0, 5, 0, 0, 5]
    assert rows["phase"].tolist() == ["endgame"] * 5


def test_every_feature_has_a_column_for_each_side(shard) -> None:
    rows = shard_rows(shard.positions, shard.games, net=None)
    assert [c for c in COLUMNS if c.endswith(("_w", "_b"))] == [
        f"{feature}_{side}" for feature in FEATURES for side in "wb"
    ]
    assert rows.loc[0, [f"{feature}_w" for feature in FEATURES]].tolist() == [
        5,
        2,
        1,
        2,
        1,
        2,
        0,
        3,
    ]
    assert rows.loc[0, [f"{feature}_b" for feature in FEATURES]].tolist() == [
        2,
        2,
        0,
        2,
        0,
        1,
        0,
        2,
    ]


def test_a_position_hash_is_its_board_and_side_to_move(shard) -> None:
    hashes = shard_rows(shard.positions, shard.games, net=None)["position_hash"].tolist()
    assert hashes[0] == hashes[3]  # the same board with white to move, in two games
    assert hashes[1] == hashes[4]  # the clocks do not count
    assert len({hashes[0], hashes[1], hashes[2]}) == 3  # another board, another side to move


def test_the_net_evaluates_every_record(shard) -> None:
    # One saturated neuron with output weight 64 is 400 centipawns for the side to move.
    values = [0] * (768 * 128 + 128 + 2 * 128 + 1)
    values[768 * 128] = 255
    values[768 * 128 + 128] = 64
    net = nnue.load_net(struct.pack(f"<{len(values)}h", *values))
    rows = shard_rows(shard.positions, shard.games, net=net)
    assert rows["eval_white"].tolist() == [400, 400, -400, 400, 400]


def test_a_line_that_is_not_a_record_is_counted_not_fatal(shard) -> None:
    rows = shard_rows(shard.positions + "not a record\n", shard.games, net=None)
    assert rows["ok"].tolist() == [True] * 5 + [False] and rows["aligned"].all()
    assert rows["game"].tolist()[:5] == [0, 0, 0, 1, 1] and rows["game"].isna().tolist()[5]
    assert not rows["kept"].tolist()[5]


def test_a_record_whose_fen_is_cut_short_is_not_a_record(shard) -> None:
    rows = shard_rows(shard.positions + "4k3/8/8/8 w | 12 | 0.5\n", shard.games, net=None)
    assert rows["ok"].tolist() == [True] * 5 + [False]


def test_a_shard_that_does_not_match_its_games_is_unaligned(shard) -> None:
    rows = shard_rows(shard.positions, shard.games.splitlines()[0] + "\n", net=None)
    assert not rows["aligned"].any() and rows["game"].isna().all() and not rows["kept"].any()
    assert rows["ok"].all() and rows["termination"].isna().all() and rows["ply"].isna().all()


def test_a_games_file_that_is_not_json_leaves_the_shard_unaligned(shard) -> None:
    rows = shard_rows(shard.positions, "{not json\n", net=None)
    assert not rows["aligned"].any() and rows["ok"].all()


def test_an_empty_shard_has_no_rows_and_every_column() -> None:
    rows = shard_rows("", "", net=None)
    assert len(rows) == 0 and tuple(rows.columns) == COLUMNS


def test_a_game_keeps_at_most_the_cap(shard) -> None:
    rows = shard_rows(shard.positions, shard.games, net=None, cap=2)
    assert rows["kept"].tolist() == [True, False, True, True, True]


def test_thinning_keeps_the_cap_evenly_spaced() -> None:
    kept = thin([0] * 6 + [1] * 2, cap=3)
    assert kept.tolist() == [True, False, True, False, True, False, True, True]


def test_folds_are_stable_and_by_worker() -> None:
    assert fold_of("r1", "t0", 17) == fold_of("r1", "t0", 17)
    assert {fold_of("r1", "t0", worker) for worker in range(200)} == set(range(FOLDS))


def test_the_schema_names_every_column_in_order() -> None:
    assert tuple(part.split()[0] for part in SCHEMA.split(", ")) == COLUMNS
