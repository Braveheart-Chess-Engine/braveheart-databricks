"""The two Spark passes: positions into bin counts, games into length and ending counts.

Serverless compute cannot cache, so each pass reads its files once and returns a small
pandas table; everything else is computed from that (analysis.py). Spark 4 and serverless
run in ANSI mode, where an out-of-range array index or a bad cast throws: parsing uses
try_element_at and try_cast, so a line that is not a record is counted, never fatal.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from .phase import ENDGAME_AT, OPENING_AT, VALUES

_GAMES_SCHEMA = "moves ARRAY<STRING>, result DOUBLE, termination STRING"


def _generation() -> Column:
    """The generation, from the shard's path (`.../gen=N/...`)."""
    return F.regexp_extract(F.col("_metadata.file_path"), r"/gen=(\d+)/", 1).cast("int")


def _count(board: Column, pieces: str) -> Column:
    return F.length(board) - F.length(F.regexp_replace(board, f"[{pieces}]", ""))


def read_positions(spark: SparkSession, paths: Sequence[str]) -> DataFrame:
    """Every line of the positions shards: gen, ok, and for records score_white, result,
    npm (non-pawn material) and phase. A line that is not a record keeps its gen with
    `ok` false, so it is counted rather than failing the run."""
    raw = spark.read.text(list(paths)).select(_generation().alias("gen"), "value")
    parts = F.split(F.col("value"), r" \| ")
    fen = F.trim(F.try_element_at(parts, F.lit(1)))
    score = F.trim(F.try_element_at(parts, F.lit(2)))
    result = F.trim(F.try_element_at(parts, F.lit(3)))
    ok = (
        (F.size(parts) == 3)
        & F.coalesce(fen != "", F.lit(False))
        & F.coalesce(score.rlike(r"^-?[0-9]+$"), F.lit(False))
        & F.coalesce(result.try_cast("double").isin(0.0, 0.5, 1.0), F.lit(False))
    )
    board = F.split(fen, " ").getItem(0)
    npm = sum(
        (_count(board, piece + piece.upper()) * value for piece, value in VALUES.items()),
        F.lit(0),
    )
    phase = (
        F.when(npm >= OPENING_AT, "opening")
        .when(npm > ENDGAME_AT, "middlegame")
        .otherwise("endgame")
    )
    return raw.select(
        "gen",
        ok.alias("ok"),
        F.when(ok, score.try_cast("int")).alias("score_white"),
        F.when(ok, result.try_cast("double")).alias("result"),
        F.when(ok, npm).alias("npm"),
        F.when(ok, phase).alias("phase"),
    )


def bin_counts(positions: DataFrame, bin_width: int = 50) -> pd.DataFrame:
    """The one aggregation the report needs: per gen, ok, phase and score bin, the
    positions and how their games ended (white's point of view)."""
    centre = F.when(
        F.col("ok"), (F.round(F.col("score_white") / bin_width) * bin_width).cast("int")
    )
    result = F.col("result")
    return (
        positions.groupBy("gen", "ok", "phase", centre.alias("bin"))
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.sum(F.when(result == 1.0, 1).otherwise(0)).alias("wins"),
            F.sum(F.when(result == 0.5, 1).otherwise(0)).alias("draws"),
            F.sum(F.when(result == 0.0, 1).otherwise(0)).alias("losses"),
        )
        .toPandas()
    )


def read_games(spark: SparkSession, paths: Sequence[str]) -> DataFrame:
    """The games of the same shards: gen, plies (engine moves after the opening) and how
    each game ended."""
    games = spark.read.schema(_GAMES_SCHEMA).json(list(paths))
    return games.select(_generation().alias("gen"), F.size("moves").alias("plies"), "termination")


def game_counts(games: DataFrame, plies_bin: int = 10) -> pd.DataFrame:
    """Per gen, ending and length bin, the games."""
    length = (F.floor(F.col("plies") / plies_bin) * plies_bin).cast("int")
    return (
        games.groupBy("gen", "termination", length.alias("plies_bin"))
        .agg(F.count(F.lit(1)).alias("n"))
        .toPandas()
    )
