import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from selfplay_analysis.shards import COLUMNS, fold_of

MIXED = "4k3/7p/3p4/3P4/2P1P3/P7/P7/4K3"
ROOKS = "r3k3/7p/3p4/3P4/2P1P3/P7/P7/R3K2R"


def _java_runs() -> bool:
    """A Java runtime that starts (macOS's /usr/bin/java is a stub without one)."""
    java = shutil.which("java")
    return (
        java is not None and subprocess.run([java, "-version"], capture_output=True).returncode == 0
    )


@pytest.fixture(scope="module")
def spark():
    # As tests/test_frames.py: required in CI, skipped locally without a full JDK.
    if os.environ.get("SPARK_TESTS") != "required" and not _java_runs():
        pytest.skip("Spark needs a Java runtime (a full JDK 17 or 21)")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("selfplay-analysis-structural-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


def _write(root: Path, gen: int, kind: str, name: str, text: str) -> str:
    path = root / f"gen={gen}" / "run=r1" / "task=t0" / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(text)
    return str(path)


def _game(white_scores: list[int], result: float, termination: str) -> str:
    scores = [score if ply % 2 == 0 else -score for ply, score in enumerate(white_scores)]
    return json.dumps(
        {
            "start_fen": f"{MIXED} w - - 0 6",
            "moves": ["a1a2"] * len(scores),
            "scores": scores,
            "result": result,
            "termination": termination,
        }
    )


WIN = _game([50, 60, 70, 80, 90, 100], 1.0, "win-adjudication")
DRAW = _game([5, 6, 7, 8], 0.5, "repetition")
LOSS = _game([-9, -8, -7], 0.0, "win-adjudication")
WIN_RECORDS = [
    f"{MIXED} w - - 0 6 | 50 | 1.0",
    f"{ROOKS} w KQq - 0 7 | 70 | 1.0",
    f"{MIXED} b - - 0 8 | 100 | 1.0",
]
DRAW_RECORDS = [f"{MIXED} w - - 0 6 | 5 | 0.5", f"{ROOKS} w KQq - 3 7 | 7 | 0.5"]
LOSS_RECORDS = [f"{MIXED} w - - 0 6 | -9 | 0.0", f"{MIXED} b - - 1 6 | -8 | 0.0"]


@pytest.fixture(scope="module")
def paths(tmp_path_factory) -> list[str]:
    """Generation 5: one shard of two games and a line that is not a record. Generation
    6: one shard of one game, and one whose games file belongs to another shard."""
    root = tmp_path_factory.mktemp("volume")
    shards = [
        (5, "t0-w017-s00000", [*WIN_RECORDS, *DRAW_RECORDS, "not a record"], [WIN, DRAW]),
        (6, "t0-w003-s00000", LOSS_RECORDS, [LOSS]),
        (6, "t0-w004-s00002", WIN_RECORDS, [DRAW]),
    ]
    out = []
    for gen, name, records, games in shards:
        out.append(_write(root, gen, "positions", f"{name}.txt.gz", "\n".join(records) + "\n"))
        _write(root, gen, "games", f"{name}.jsonl.gz", "\n".join(games) + "\n")
    return out


def test_read_shards_gives_a_row_per_line_with_its_provenance(spark, paths) -> None:
    from selfplay_analysis.structural_frames import read_shards

    table = read_shards(spark, paths, net_bytes=None)
    assert table.columns == ["gen", "file", "run", "task", "worker", "fold", *COLUMNS]
    rows = table.collect()
    assert len(rows) == 6 + 2 + 3
    assert {(r["gen"], r["run"], r["task"], r["worker"]) for r in rows} == {
        (5, "r1", "t0", 17),
        (6, "r1", "t0", 3),
        (6, "r1", "t0", 4),
    }
    assert all(r["fold"] == fold_of("r1", "t0", r["worker"]) for r in rows)
    first = next(r for r in rows if r["worker"] == 17 and r["score_white"] == 50)
    assert first["file"].endswith("/gen=5/run=r1/task=t0/positions/t0-w017-s00000.txt.gz")
    assert (first["game"], first["ply"], first["termination"]) == (0, 0, "win-adjudication")
    assert (first["isolated_w"], first["passed_b"], first["eval_white"]) == (2, 1, None)


def test_read_shards_evaluates_with_the_net_it_is_given(spark, paths) -> None:
    import struct

    from selfplay_analysis.structural_frames import read_shards

    values = [0] * (768 * 128 + 128 + 2 * 128 + 1)
    values[768 * 128] = 255
    values[768 * 128 + 128] = 64
    net_bytes = struct.pack(f"<{len(values)}h", *values)
    rows = read_shards(spark, paths[:1], net_bytes=net_bytes).collect()
    assert sorted(r["eval_white"] for r in rows if r["ok"]) == [-400, 400, 400, 400, 400]


def test_files_that_do_not_decompress_are_not_fatal(spark, tmp_path: Path) -> None:
    from selfplay_analysis.structural_frames import read_shards

    records = "\n".join(LOSS_RECORDS) + "\n"
    paths = []
    for worker, good_positions, good_games in ((7, False, True), (8, True, True), (9, True, False)):
        name = f"t0-w00{worker}-s00000"
        paths.append(_write(tmp_path, 6, "positions", f"{name}.txt.gz", records))
        games = Path(_write(tmp_path, 6, "games", f"{name}.jsonl.gz", LOSS + "\n"))
        if not good_positions:
            Path(paths[-1]).write_bytes(b"not gzip")
        if not good_games:
            games.write_bytes(b"not gzip")
    rows = read_shards(spark, paths, net_bytes=None).collect()
    # Worker 7's positions give no rows (the manifest check reports the shard); worker 9's
    # records are read, with no games to assign them to.
    assert {(r["worker"], r["aligned"]) for r in rows} == {(8, True), (9, False)}
    assert sum(r["ok"] for r in rows) == 4


def test_the_summary_counts_files_endings_and_features(spark, paths) -> None:
    from selfplay_analysis.structural_frames import read_shards, summary

    table = summary(read_shards(spark, paths, net_bytes=None))
    files = table[table["kind"] == "file"].sort_values("file").reset_index(drop=True)
    assert files["gen"].tolist() == [5, 6, 6]
    assert files["n"].tolist() == [6, 2, 3]  # lines
    assert files["ok"].tolist() == [5, 2, 3]
    assert files["aligned"].tolist() == [1, 1, 0]
    assert files["games"].tolist() == [2, 1, 0]
    assert files["kept"].tolist() == [5, 2, 0]
    endings = table[table["kind"] == "ending"].set_index(["gen", "termination"])
    assert endings.loc[(5, "win-adjudication"), ["n", "games"]].tolist() == [3, 1]
    assert endings.loc[(5, "repetition"), ["n", "games"]].tolist() == [2, 1]
    assert endings.loc[(6, "win-adjudication"), ["n", "games"]].tolist() == [2, 1]
    features = table[table["kind"] == "feature"].set_index(["gen", "phase", "feature", "value"])
    # MIXED and ROOKS share their pawns: white has one more passed pawn and as many isolated.
    assert features.loc[(5, "endgame", "passed", 1), "n"] == 5
    assert features.loc[(5, "endgame", "isolated", 0), "n"] == 5
    assert features.loc[(6, "endgame", "passed", 1), "n"] == 2  # the unaligned shard is left out
    assert set(features.index.get_level_values("feature")) == {
        "isolated",
        "doubled",
        "islands",
        "connected",
        "passed",
        "protected_passed",
        "passed_rank",
    }


def test_the_summarys_files_check_against_the_manifests(spark, paths, tmp_path: Path) -> None:
    from selfplay_analysis import analysis
    from selfplay_analysis.frames import read_manifests
    from selfplay_analysis.sampling import manifest_path_for
    from selfplay_analysis.structural_frames import read_shards, summary

    manifests = []
    for path, positions in zip(paths, [6, 2, 4], strict=True):
        manifest = Path(manifest_path_for(path))
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"positions": positions}))
        manifests.append(str(manifest))
    table = summary(read_shards(spark, paths, net_bytes=None))
    files = table[table["kind"] == "file"]
    lines = files[["gen", "file", "n"]].rename(columns={"n": "lines"})
    complete = analysis.completeness(lines, read_manifests(spark, manifests))
    assert (complete["short"], complete["unreadable"], complete["missing"]) == (1, 0, 1)


def test_model_rows_are_the_kept_rows_of_aligned_shards(spark, paths) -> None:
    from selfplay_analysis.structural_frames import model_rows, read_shards

    rows, share = model_rows(read_shards(spark, paths, net_bytes=None), kept=7, max_rows=10**6)
    assert share == 1.0 and len(rows) == 7
    assert "file" not in rows.columns and rows["game_id"].nunique() == 3
    assert sorted(rows["gen"].unique()) == [5, 6]
    assert set(rows["pawns_d"]) == {3} and set(rows["passed_d"]) == {1}
    assert set(rows["isolated_d"]) == {0} and set(rows["passed_rank_d"]) == {1}
    for column in ("fold", "ply", "stm", "result", "termination", "npm_diff", "phase"):
        assert column in rows.columns
    assert "position_hash" in rows.columns and "eval_white" in rows.columns


def test_model_rows_over_the_cap_are_whole_games_and_the_same_every_time(spark, paths) -> None:
    from selfplay_analysis.structural_frames import model_rows, read_shards

    table = read_shards(spark, paths, net_bytes=None)
    everything, _ = model_rows(table, kept=7, max_rows=10**6)
    sizes = everything.groupby("game_id").size()
    first, share = model_rows(table, kept=7, max_rows=3)
    again, _ = model_rows(table, kept=7, max_rows=3)
    assert share == pytest.approx(3 / 7)
    assert sorted(first["position_hash"]) == sorted(again["position_hash"])
    assert (first.groupby("game_id").size() == sizes[first["game_id"].unique()]).all()
