# Databricks notebook source
# MAGIC %md
# MAGIC # Do the scores predict the results?
# MAGIC
# MAGIC Braveheart's nets train on `λ × result + (1 − λ) × sigmoid(score / 400)`: every
# MAGIC self-play position carries the search's score and how its game ended. Were the
# MAGIC scores calibrated, white would score `sigmoid(s / 400)` on average among positions
# MAGIC scored `s`. This report checks that, generation by generation, on
# MAGIC braveheart-selfplay's positions, read through the volume
# MAGIC `braveheart_selfplay.positions.files` (read-only).
# MAGIC
# MAGIC **Parameters** (the widgets above, or the job's): `shards_per_generation`, 24 by
# MAGIC default (about 600,000 positions a generation; `all` reads every shard, about 3.2
# MAGIC billion positions); `generations`; `label_set`, only `original` for now.

# COMMAND ----------

# MAGIC %pip install --quiet plotly==7.1.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import glob
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from selfplay_analysis import analysis, charts
from selfplay_analysis.frames import bin_counts, game_counts, read_games, read_positions
from selfplay_analysis.sampling import check_label_set, choose_shards, games_path_for

VOLUME = "/Volumes/braveheart_selfplay/positions/files/selfplay/v1"

dbutils.widgets.text("shards_per_generation", "24", "Shards per generation, or all")
dbutils.widgets.text("generations", "0,1,2,3,4,5,6", "Generations")
dbutils.widgets.text("label_set", "original", "Label set")

shards = dbutils.widgets.get("shards_per_generation").strip().lower()
per_generation = None if shards == "all" else int(shards)
generations = [int(g) for g in dbutils.widgets.get("generations").split(",") if g.strip()]
check_label_set(dbutils.widgets.get("label_set").strip())

# COMMAND ----------

# Every positions shard of the chosen generations, then the sample.
every_shard = {
    gen: sorted(glob.glob(f"{VOLUME}/gen={gen}/run=*/task=*/positions/*.txt.gz"))
    for gen in generations
}
picked = choose_shards(every_shard, per_generation, seed=0)
positions_paths = [path for paths in picked.values() for path in paths]
games_paths = [games_path_for(path) for path in positions_paths]
total = sum(len(paths) for paths in every_shard.values())
print(f"{len(positions_paths):,} of {total:,} shards, generations {generations}")

# COMMAND ----------

# One pass over the positions and one over the games: serverless compute cannot
# cache, so each pass reads its files once, and everything below comes from two
# small tables.
try:
    spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
except Exception as exc:  # not every compute lets a notebook set it
    print(f"note: a corrupt file will fail the run ({type(exc).__name__})")
bins = bin_counts(read_positions(spark, positions_paths))
games = game_counts(read_games(spark, games_paths))

positions_read = int(bins.loc[bins["ok"].astype(bool), "n"].sum())
skipped = analysis.skipped_lines(bins)
if per_generation is None:
    sample_label = f"every shard: {positions_read:,} positions"
else:
    sample_label = f"sample: {per_generation} shards a generation, {positions_read:,} positions"
if skipped:
    print(f"lines that were not records, skipped: {skipped}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The dataset at a glance

# COMMAND ----------

overview = {
    "positions": positions_read,
    "games": int(games["n"].sum()),
    "generations": len(generations),
    "share": len(positions_paths) / total,
}
charts.overview_tiles(overview, sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Do scores predict results?
# MAGIC
# MAGIC Each line is a generation: white's mean result among positions with that score (a
# MAGIC draw counts half). The dashed white line is what the trainer assumes. A curve
# MAGIC flatter than the line means the scores overstate how decisive positions are; a
# MAGIC steeper one, that they understate it.

# COMMAND ----------

charts.calibration_curves(analysis.calibration_table(bins), sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Is 400 still the right scale?
# MAGIC
# MAGIC The scale whose sigmoid best predicts each generation's results (least log loss).
# MAGIC Above 400, a centipawn is worth less than the trainer assumes; below 400, more.

# COMMAND ----------

scales = analysis.fitted_scales(bins)
charts.scale_bars(scales, sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where does the fit break?
# MAGIC
# MAGIC White's mean result minus `sigmoid(score / 400)`, by score and phase (non-pawn
# MAGIC material: opening 50 or more, endgame 20 or less). Blue: white does better than its
# MAGIC score says; red: worse. The menu picks a generation.

# COMMAND ----------

by_phase = analysis.calibration_table(bins, by_phase=True)
charts.phase_heatmap(by_phase, sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## A portrait of each generation

# COMMAND ----------

charts.results_mix(analysis.results_mix(bins), sample_label).show()
charts.score_ridgeline(analysis.score_histogram(bins), sample_label).show()
charts.phase_mix(analysis.phase_mix(bins), sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How the games went
# MAGIC
# MAGIC From the games files of the same shards: engine plies after the random opening,
# MAGIC and how each game ended.

# COMMAND ----------

charts.game_lengths(analysis.game_lengths(games), sample_label).show()
charts.terminations(analysis.terminations(games), sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## In numbers

# COMMAND ----------

display(
    scales.assign(scale=scales["scale"].round(0)).rename(
        columns={"gen": "generation", "scale": "best scale", "n": "positions in the fit"}
    )
)
