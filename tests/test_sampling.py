import os

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


def _tree(root, shards: dict[str, bool]) -> str:
    """A volume with positions shards; True means the shard's manifest exists."""
    volume = root / "v1"
    for name, complete in shards.items():
        task = volume / "gen=6" / "run=r1" / "task=t0"
        (task / "positions").mkdir(parents=True, exist_ok=True)
        (task / "positions" / f"{name}.txt.gz").write_bytes(b"")
        if complete:
            (task / "manifests").mkdir(exist_ok=True)
            (task / "manifests" / f"{name}.json").write_text("{}")
    return str(volume)


def test_discover_shards_finds_complete_shards_by_their_manifests(tmp_path) -> None:
    from selfplay_analysis.sampling import discover_shards

    volume = _tree(
        tmp_path, {"t0-w000-s00000": True, "t0-w000-s00001": True, "t0-w000-s00002": False}
    )
    found = discover_shards(volume, [6])
    assert [p.rsplit("/", 1)[1] for p in found[6]] == [
        "t0-w000-s00000.txt.gz",
        "t0-w000-s00001.txt.gz",
    ]
    assert all("/positions/" in p for p in found[6])


def test_discover_shards_raises_when_the_volume_cannot_be_read(tmp_path) -> None:
    from selfplay_analysis.sampling import discover_shards

    volume = _tree(tmp_path, {"t0-w000-s00000": True})
    os.chmod(volume, 0)
    try:
        with pytest.raises(PermissionError):
            discover_shards(volume, [6])
    finally:
        os.chmod(volume, 0o755)
    with pytest.raises(FileNotFoundError):
        discover_shards(str(tmp_path / "missing"), [6])


def test_a_shards_manifest_and_positions_paths_map_both_ways() -> None:
    from selfplay_analysis.sampling import manifest_path_for, positions_path_for

    shard = f"{ROOT}/gen=6/run=r/task=t0"
    assert (
        manifest_path_for(f"{shard}/positions/t0-w000-s00000.txt.gz")
        == f"{shard}/manifests/t0-w000-s00000.json"
    )
    assert (
        positions_path_for(f"{shard}/manifests/t0-w000-s00000.json")
        == f"{shard}/positions/t0-w000-s00000.txt.gz"
    )
