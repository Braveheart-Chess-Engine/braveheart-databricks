import pytest

from selfplay_analysis.sampling import (
    LABEL_SETS,
    check_label_set,
    choose_shards,
    games_path_for,
    generation_of,
)

ROOT = "/Volumes/braveheart_selfplay/positions/files/selfplay/v1"


def _paths(gen: int, n: int) -> list[str]:
    return [f"{ROOT}/gen={gen}/run=r1/task=t0/positions/t0-w000-s{i:05d}.txt.gz" for i in range(n)]


def test_choose_shards_is_deterministic_capped_and_sorted() -> None:
    paths = {0: _paths(0, 100), 6: _paths(6, 10)}
    picked = choose_shards(paths, 24, seed=0)
    assert picked == choose_shards(paths, 24, seed=0)
    assert len(picked[0]) == 24
    assert picked[0] == sorted(picked[0])
    assert set(picked[0]) <= set(paths[0])
    assert picked[6] == sorted(paths[6])  # fewer than asked: every one


def test_every_shard_when_none_is_asked() -> None:
    paths = {0: _paths(0, 30)}
    assert choose_shards(paths, None) == {0: sorted(paths[0])}


def test_a_generation_without_shards_is_refused() -> None:
    with pytest.raises(ValueError, match="7"):
        choose_shards({0: _paths(0, 3), 7: []}, 24)


def test_a_sample_of_no_shards_is_refused() -> None:
    with pytest.raises(ValueError):
        choose_shards({0: _paths(0, 3)}, 0)


def test_games_path_for_a_positions_shard() -> None:
    shard = f"{ROOT}/gen=6/run=r/task=t0"
    assert games_path_for(f"{shard}/positions/t0-w000-s00000.txt.gz") == (
        f"{shard}/games/t0-w000-s00000.jsonl.gz"
    )


def test_generation_of_a_path() -> None:
    assert generation_of(f"{ROOT}/gen=6/run=20261108-1/task=t0/positions/x.txt.gz") == 6


def test_only_the_original_labels_are_accepted() -> None:
    assert LABEL_SETS == ("original",)
    assert check_label_set("original") == "original"
    with pytest.raises(ValueError, match="only the original labels"):
        check_label_set("net7-l050-r41363264869-n5000")
