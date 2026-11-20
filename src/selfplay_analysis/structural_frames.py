"""The structural analysis's two Spark passes: every sampled shard through shards.py, once
for the counts the report shows and once for the rows its models are fitted on.

A shard is read whole (`binaryFile`), its positions and games files together, because the
game a record belongs to is known only from the order of its file and from its games
file. Serverless compute cannot cache, so each pass runs the per-shard Python again and
returns a pandas table small enough for the driver.
"""

from __future__ import annotations

import gzip
import re
import zlib
from collections.abc import Iterator, Sequence

import pandas as pd
from pyspark import cloudpickle
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

import selfplay_analysis

from .nnue import load_net
from .registered import MAX_MODEL_ROWS, MAX_POSITIONS_PER_GAME
from .sampling import games_path_for
from .shards import COLUMNS, SCHEMA, fold_of, shard_rows

#: The structure features the model uses, as white's count minus black's.
STRUCTURE = (
    "isolated",
    "doubled",
    "islands",
    "connected",
    "passed",
    "protected_passed",
    "passed_rank",
)
_PROVENANCE = "gen INT, file STRING, run STRING, task STRING, worker INT, fold TINYINT"
_SHARD = re.compile(
    r"/gen=(?P<gen>\d+)/run=(?P<run>[^/]+)/task=(?P<task>[^/]+)/positions/[^/]*-w(?P<worker>\d+)-s\d+\.txt\.gz$"
)
#: The part of a positions or games path that names its shard.
_KEY = r"(gen=\d+/run=[^/]+/task=[^/]+)/(?:positions|games)/([^/]+?)\.(?:txt|jsonl)\.gz$"


def _key(path: Column) -> Column:
    return F.concat_ws("/", F.regexp_extract(path, _KEY, 1), F.regexp_extract(path, _KEY, 2))


def _text(content: bytes | None) -> str | None:
    """A gzipped file's text, or None when it is missing or does not decompress."""
    if content is None:
        return None
    try:
        return gzip.decompress(content).decode("utf-8", errors="replace")
    except (OSError, EOFError, zlib.error):
        return None


def read_shards(
    spark: SparkSession,
    positions_paths: Sequence[str],
    net_bytes: bytes | None,
    cap: int = MAX_POSITIONS_PER_GAME,
) -> DataFrame:
    """A row per line of the positions shards: gen, file, run, task, worker and fold (of
    the worker group), then shards.py's columns. `net_bytes` is the net file, evaluated in
    each task; None leaves `eval_white` null. A positions file that does not decompress
    gives no rows (the manifest check reports it), and a games file that does not leaves
    its shard unaligned. A games file that is not there at all fails the read, naming it:
    a shard's manifest is uploaded last, so its files exist."""
    path = F.col("_metadata.file_path")
    positions = (
        spark.read.format("binaryFile")
        .load(list(positions_paths))
        .select(_key(path).alias("shard"), path.alias("file"), F.col("content").alias("positions"))
    )
    games = (
        spark.read.format("binaryFile")
        .load([games_path_for(p) for p in positions_paths])
        .select(_key(path).alias("shard"), F.col("content").alias("games"))
    )
    shards = positions.join(games, "shard", "left").repartition(
        max(1, min(len(positions_paths), 64)), "shard"
    )

    def rows(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        net = load_net(net_bytes) if net_bytes is not None else None
        for batch in batches:
            for file, positions_gz, games_gz in zip(
                batch["file"], batch["positions"], batch["games"], strict=True
            ):
                text = _text(positions_gz)
                found = _SHARD.search(file)
                if text is None or found is None:
                    continue
                frame = shard_rows(text, _text(games_gz) or "", net, cap)
                run, task, worker = found["run"], found["task"], int(found["worker"])
                frame.insert(0, "fold", fold_of(run, task, worker))
                frame.insert(0, "worker", worker)
                frame.insert(0, "task", task)
                frame.insert(0, "run", run)
                frame.insert(0, "file", file)
                frame.insert(0, "gen", int(found["gen"]))
                yield frame

    # The Python that runs each task does not have this package to import (the notebook
    # puts `src` on the driver's path only), so its functions travel with the task.
    cloudpickle.register_pickle_by_value(selfplay_analysis)
    return shards.select("file", "positions", "games").mapInPandas(
        rows, schema=f"{_PROVENANCE}, {SCHEMA}"
    )


def _difference(feature: str) -> Column:
    return (F.col(f"{feature}_w") - F.col(f"{feature}_b")).cast("int")


def summary(table: DataFrame) -> pd.DataFrame:
    """Everything the report counts, in one scan, as rows of three kinds:

    - `file`: per gen and file, the lines read (`n`), those that are records (`ok`),
      whether the shard was assigned to its games (`aligned`, 1 or 0), its games and the
      rows that survive thinning (`kept`);
    - `ending`: per gen and termination, the positions (`n`) and the games;
    - `feature`: per gen, phase, structure feature and its white-minus-black `value`, the
      positions (`n`).

    Endings and features count the records of aligned shards, before thinning."""
    usable = F.col("ok") & F.col("aligned")
    null_string, null_int = F.lit(None).cast("string"), F.lit(None).cast("int")
    kinds = F.array(
        F.struct(
            F.lit("file").alias("kind"),
            F.col("file").alias("file"),
            null_string.alias("phase"),
            null_string.alias("termination"),
            null_string.alias("feature"),
            null_int.alias("value"),
        ),
        F.struct(
            F.when(usable, "ending").alias("kind"),
            null_string.alias("file"),
            null_string.alias("phase"),
            F.col("termination").alias("termination"),
            null_string.alias("feature"),
            null_int.alias("value"),
        ),
        *(
            F.struct(
                F.when(usable, "feature").alias("kind"),
                null_string.alias("file"),
                F.col("phase").alias("phase"),
                null_string.alias("termination"),
                F.lit(feature).alias("feature"),
                _difference(feature).alias("value"),
            )
            for feature in STRUCTURE
        ),
    )
    game = F.when(usable, F.concat_ws("#", F.col("file"), F.col("game")))
    counted = (
        table.select(
            "gen",
            F.col("ok").cast("int").alias("ok"),
            F.col("aligned").cast("int").alias("aligned"),
            F.col("kept").cast("int").alias("kept"),
            game.alias("game"),
            F.explode(kinds).alias("row"),
        )
        .where(F.col("row.kind").isNotNull())
        .groupBy(
            F.col("row.kind").alias("kind"),
            "gen",
            F.col("row.file").alias("file"),
            F.col("row.phase").alias("phase"),
            F.col("row.termination").alias("termination"),
            F.col("row.feature").alias("feature"),
            F.col("row.value").alias("value"),
        )
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.sum("ok").alias("ok"),
            F.max("aligned").alias("aligned"),
            F.countDistinct("game").alias("games"),
            F.sum("kept").alias("kept"),
        )
    )
    return counted.toPandas()


def model_rows(
    table: DataFrame, kept: int, max_rows: int = MAX_MODEL_ROWS
) -> tuple[pd.DataFrame, float]:
    """The rows the models are fitted on, and the share of games they hold: the records of
    aligned shards that survive thinning, with `game_id` (a hash of the file and the game)
    and each structure feature and the pawn count as white's minus black's (`<feature>_d`,
    `pawns_d`). `kept` is how many such rows there are (the summary's count): beyond
    `max_rows`, whole games are sampled by their `game_id`, the same ones every time."""
    share = 1.0 if kept <= max_rows else max_rows / kept
    rows = table.where(F.col("ok") & F.col("aligned") & F.col("kept")).select(
        "gen",
        "fold",
        F.xxhash64("file", "game").alias("game_id"),
        "ply",
        "stm",
        "score_white",
        "result",
        "termination",
        "npm_diff",
        "phase",
        "position_hash",
        "eval_white",
        _difference("pawns").alias("pawns_d"),
        *(_difference(feature).alias(f"{feature}_d") for feature in STRUCTURE),
    )
    if share < 1.0:
        rows = rows.where(F.pmod(F.col("game_id"), F.lit(1_000_000)) < int(share * 1_000_000))
    return rows.toPandas(), share


__all__ = ["COLUMNS", "STRUCTURE", "model_rows", "read_shards", "summary"]
