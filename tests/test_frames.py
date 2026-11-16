import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ROOK_ENDING = "4k2r/8/8/8/8/8/8/R3K3 w - - 0 1"


def _java_runs() -> bool:
    """A Java runtime that starts (macOS's /usr/bin/java is a stub without one)."""
    java = shutil.which("java")
    return (
        java is not None and subprocess.run([java, "-version"], capture_output=True).returncode == 0
    )


@pytest.fixture(scope="module")
def spark():
    # CI sets SPARK_TESTS=required, so these never pass there by skipping; locally they
    # skip without a Java runtime (Spark 4 needs a full JDK, with jdk.incubator.vector).
    if os.environ.get("SPARK_TESTS") != "required" and not _java_runs():
        pytest.skip("Spark needs a Java runtime (a full JDK 17 or 21)")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("selfplay-analysis-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


def _shard(root: Path, gen: int, kind: str, name: str, lines: list[str]) -> str:
    path = root / f"gen={gen}" / "run=r1" / "task=t0" / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def test_read_positions_parses_and_measures_phase(spark, tmp_path: Path) -> None:
    from selfplay_analysis.frames import read_positions

    path = _shard(
        tmp_path,
        6,
        "positions",
        "t0-w000-s00000.txt.gz",
        [f"{START} | 20 | 0.5", f"{ROOK_ENDING} | -310 | 0.0"],
    )
    rows = {r["score_white"]: r for r in read_positions(spark, [path]).collect()}
    assert rows[20]["gen"] == 6 and rows[20]["ok"] and rows[20]["result"] == 0.5
    assert rows[20]["npm"] == 62 and rows[20]["phase"] == "opening"
    assert rows[-310]["npm"] == 10 and rows[-310]["phase"] == "endgame"


def test_malformed_lines_are_flagged_not_fatal(spark, tmp_path: Path) -> None:
    from selfplay_analysis.frames import read_positions

    path = _shard(
        tmp_path,
        6,
        "positions",
        "t0-w000-s00001.txt.gz",
        [f"{START} | 20 | 0.5", "not a record", f"{START} | 3.5 | 0.5", f"{START} | 20 | abc"],
    )
    rows = read_positions(spark, [path]).collect()
    assert sorted(r["ok"] for r in rows) == [False, False, False, True]
    assert all(r["gen"] == 6 for r in rows)


def test_bin_counts_aggregate_generation_phase_and_bin(spark, tmp_path: Path) -> None:
    from selfplay_analysis.frames import bin_counts, read_positions

    path = _shard(
        tmp_path,
        3,
        "positions",
        "t0-w000-s00002.txt.gz",
        [f"{START} | 24 | 1.0", f"{START} | -20 | 0.5", f"{START} | 30 | 0.0", "bad"],
    )
    bins = bin_counts(read_positions(spark, [path]))
    ok = bins[bins["ok"]].set_index("bin")
    assert ok.loc[0, "n"] == 2 and ok.loc[0, "wins"] == 1 and ok.loc[0, "draws"] == 1
    assert ok.loc[50, "n"] == 1 and ok.loc[50, "losses"] == 1
    assert bins[~bins["ok"]]["n"].tolist() == [1]


def test_read_games_counts_lengths_and_terminations(spark, tmp_path: Path) -> None:
    from selfplay_analysis.frames import game_counts, read_games

    games = [
        {
            "start_fen": START,
            "moves": ["e2e4"] * 83,
            "scores": [0] * 83,
            "result": 1.0,
            "termination": "checkmate",
            "samples": [],
        },
        {
            "start_fen": START,
            "moves": ["d2d4"] * 120,
            "scores": [0] * 120,
            "result": 0.5,
            "termination": "repetition",
            "samples": [],
        },
    ]
    path = _shard(tmp_path, 5, "games", "t0-w000-s00000.jsonl.gz", [json.dumps(g) for g in games])
    counts = game_counts(read_games(spark, [path])).set_index("termination")
    assert counts.loc["checkmate", "plies_bin"] == 80 and counts.loc["checkmate", "gen"] == 5
    assert counts.loc["repetition", "plies_bin"] == 120 and counts["n"].sum() == 2
