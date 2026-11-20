# Databricks notebook source
# MAGIC %md
# MAGIC # Does the net misjudge pawn structure?
# MAGIC
# MAGIC Braveheart's net sees pawn structure only through piece-square inputs. This report
# MAGIC asks whether its **raw static evaluation** is wrong by structure: among positions it
# MAGIC scores the same, does the side with, say, the extra passed pawn go on to score
# MAGIC better than the evaluation says? It reads braveheart-selfplay's positions through
# MAGIC the volume `braveheart_selfplay.positions.files` (read-only) and evaluates each with
# MAGIC `net8x160-l050-r43390801430`, in a port of the engine's inference that is checked
# MAGIC against the engine before any position is read.
# MAGIC
# MAGIC The answer is one of three, by a rule registered before any result
# MAGIC ([the research note](https://github.com/Braveheart-Chess/braveheart-databricks/blob/main/docs/nnue-structural-research.md)):
# MAGIC **GO** (a pawn-pair net comparison is worth its cost), **NO-GO** (the data could have
# MAGIC shown an effect of the size that matters, and did not) or **INCONCLUSIVE**. An effect
# MAGIC is in centipawns of evaluation per unit of a feature's white-minus-black difference,
# MAGIC in one phase; the rule asks for one of at least `min_effect_cp` that both generations
# MAGIC agree on, for shuffled features to show none, and for the sample to be large enough
# MAGIC to have seen one.
# MAGIC
# MAGIC **Parameters** (the widgets above, or the job's): `generations`, each analysed on its
# MAGIC own; `shards_per_generation`, 672 by default (about 33.6 million positions a pass);
# MAGIC `sampling_seed`; `min_effect_cp`, registered at 15 and not to be changed after a run;
# MAGIC `max_positions_per_game`; `exclude_draw_adjudicated`, the sensitivity run; `net_path`,
# MAGIC a copy of the net file in the workspace (empty downloads it from the engine's
# MAGIC repository).

# COMMAND ----------

# MAGIC %pip install --quiet plotly==7.1.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from selfplay_analysis import analysis, charts, nnue, registered, structural_charts
from selfplay_analysis.frames import read_manifests
from selfplay_analysis.sampling import choose_shards, discover_shards, manifest_path_for
from selfplay_analysis.structural_frames import model_rows, read_shards, summary
from selfplay_analysis.structural_model import (
    cross_fitted,
    difference_shares,
    duplicate_share,
    effects,
    gain,
    shuffle_structure,
)
from selfplay_analysis.verdict import decide

VOLUME = "/Volumes/braveheart_selfplay/positions/files/selfplay/v1"

dbutils.widgets.text("generations", "5,6", "Generations")
dbutils.widgets.text("shards_per_generation", "672", "Shards per generation, or all")
dbutils.widgets.text("sampling_seed", "0", "Sampling seed")
dbutils.widgets.text("min_effect_cp", "15", "Smallest effect that matters (cp)")
dbutils.widgets.text("max_positions_per_game", "24", "Positions kept per game")
dbutils.widgets.text("exclude_draw_adjudicated", "false", "Leave out draw-adjudicated games")
dbutils.widgets.text("net_path", "", "Net file in the workspace (empty: download)")

generations = [int(g) for g in dbutils.widgets.get("generations").split(",") if g.strip()]
shards = dbutils.widgets.get("shards_per_generation").strip().lower()
per_generation = None if shards == "all" else int(shards)
sampling_seed = int(dbutils.widgets.get("sampling_seed"))
min_effect_cp = float(dbutils.widgets.get("min_effect_cp"))
cap = int(dbutils.widgets.get("max_positions_per_game"))
exclude_draws = dbutils.widgets.get("exclude_draw_adjudicated").strip().lower() == "true"
net_path = dbutils.widgets.get("net_path").strip()

# COMMAND ----------

# The net, and the proof that this port evaluates as the engine does: its SHA-256, then a
# set of positions the engine itself evaluated. Either failing stops the run here, before
# a position is read.
if net_path:
    with open(net_path, "rb") as f:
        net_bytes = f.read()
else:
    try:
        net_bytes = nnue.download_net()
    except OSError as exc:
        raise RuntimeError(
            f"could not download the net from {nnue.NET_URL} ({exc}): upload the file to "
            "the workspace and give its path as net_path"
        ) from exc
net = nnue.load_net(net_bytes, sha256=nnue.NET_SHA256)
print(f"{nnue.NET_ID}: {nnue.check_golden(net):,} positions evaluated exactly as the engine does")

# COMMAND ----------

# Every complete shard (its manifest exists: the producer uploads it last), then the
# sample. The volume is listed first, so a missing grant fails here, naming it.
every_shard = discover_shards(VOLUME, generations)
picked = choose_shards(every_shard, per_generation, seed=sampling_seed)
positions_paths = [path for paths in picked.values() for path in paths]
total = sum(len(paths) for paths in every_shard.values())
print(f"{len(positions_paths):,} of {total:,} shards, generations {generations}")

# COMMAND ----------

# Pass 1: every sampled shard through shards.py (its games, the net's evaluation, both
# sides' pawn structure), counted. Serverless compute cannot cache, so the second pass
# below reads the shards again for the rows the models are fitted on.
table = read_shards(spark, positions_paths, net_bytes, cap=cap)
counts = summary(table)
files = counts[counts["kind"] == "file"]
lines = files[["gen", "file", "n"]].rename(columns={"n": "lines"})
expected = read_manifests(spark, [manifest_path_for(path) for path in positions_paths])
complete = analysis.completeness(lines, expected)
kept = int(files.loc[files["aligned"] == 1, "kept"].sum())
rows, games_share = model_rows(table, kept)

positions_read = int(files["ok"].sum())
games_read = int(files["games"].sum())
scope = "every shard" if per_generation is None else f"sample: {per_generation} shards a generation"
sample_label = f"{scope}, {positions_read:,} positions, {len(rows):,} in the model"
if exclude_draws:
    sample_label += ", draw-adjudicated games left out"
games_label = f"{scope}, {games_read:,} games"
print(f"shards against their manifests: {complete}")
if games_share < 1.0:
    print(f"the model holds {games_share:.1%} of the games: the sample is over its row limit")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the report read

# COMMAND ----------

overview = {
    "positions": positions_read,
    "games": games_read,
    "shards": len(positions_paths),
    "total_shards": total,
    "model_rows": len(rows),
    "unaligned": int((files["aligned"] == 0).sum()),
    "skipped": int((files["n"] - files["ok"]).sum()),
    **{key: complete[key] for key in ("short", "unreadable", "missing")},
}
structural_charts.coverage_tiles(overview, sample_label).show()
endings = counts[counts["kind"] == "ending"][["gen", "termination", "games"]]
charts.terminations(
    analysis.terminations(endings.rename(columns={"games": "n"})), games_label
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How often the sides differ
# MAGIC
# MAGIC A feature can only explain a result where the two sides differ in it. A feature in a
# MAGIC phase is **eligible** when they do in at least 5% of that phase's positions, in every
# MAGIC generation; the rule looks at eligible cells only.

# COMMAND ----------

shares = difference_shares(counts)
structural_charts.structure_shares(shares, registered.MIN_SHARE, sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## The models
# MAGIC
# MAGIC Per generation, a logistic model of the game's result from the evaluation, the
# MAGIC material and the side to move, with and without the seven structure differences,
# MAGIC each with its own coefficient per phase. The effects come from one fit on all of a
# MAGIC generation's games with standard errors clustered by game; the control is the same
# MAGIC fit on features shuffled across positions within cells of phase and evaluation.

# COMMAND ----------

effects_by_generation, shuffled_by_generation, gains, duplicates = {}, {}, {}, {}
for gen in generations:
    part = rows[rows["gen"] == gen]
    if exclude_draws:
        part = part[part["termination"] != "draw-adjudication"]
    part = part.reset_index(drop=True)
    mixed = shuffle_structure(part, seed=sampling_seed)
    effects_by_generation[gen] = effects(part, min_effect_cp)
    shuffled_by_generation[gen] = effects(mixed, min_effect_cp)
    # The baseline has no structure term, so it predicts the shuffled rows as it does the real.
    base = cross_fitted(part, structure=False)
    gains[gen] = {
        "structure": gain(part, base, cross_fitted(part, structure=True)),
        "shuffled": gain(mixed, base, cross_fitted(mixed, structure=True)),
    }
    duplicates[gen] = duplicate_share(part)
    print(
        f"generation {gen}: {part['game_id'].nunique():,} games, {len(part):,} positions; "
        f"{duplicates[gen]:.2%} of positions also appear in another fold"
    )
verdict = decide(effects_by_generation, shuffled_by_generation, min_effect_cp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Could the report have seen an effect?
# MAGIC
# MAGIC Before the effects: the chance each eligible cell would have been detected in every
# MAGIC generation at once, were its effect exactly the smallest that matters. If any is
# MAGIC under 80%, the run decides nothing, whatever the effects below look like: raise
# MAGIC `shards_per_generation`.

# COMMAND ----------

structural_charts.power_grid(verdict, registered.POWER_FLOOR, sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the evaluation misses
# MAGIC
# MAGIC Each cell's effect with its 95% interval. Right of zero, the side with more of the
# MAGIC feature scores better than the evaluation says: the net undervalues it. The second
# MAGIC chart is the control: the same analysis on shuffled features, where no effect can be.

# COMMAND ----------

structural_charts.effect_forest(
    effects_by_generation, verdict.cells, min_effect_cp, sample_label
).show()
structural_charts.effect_forest(
    shuffled_by_generation,
    verdict.cells,
    min_effect_cp,
    sample_label,
    title="The control: the same analysis on shuffled features",
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Do the features predict results better?
# MAGIC
# MAGIC The drop in log loss from adding the structure terms, on predictions each position
# MAGIC gets from a model fitted without its worker group. Shown for scale, and not a
# MAGIC condition of the rule: predicting better with 21 more coefficients takes several
# MAGIC times the evidence that measuring one of them does.

# COMMAND ----------

structural_charts.gain_intervals(gains, sample_label).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## The verdict

# COMMAND ----------

structural_charts.verdict_banner(verdict, min_effect_cp, sample_label).show()

# COMMAND ----------

# What the run supports, said from the verdict and never typed by hand.
met = verdict.cells[verdict.cells["meets_size"]]
weak = verdict.cells[
    verdict.cells["eligible"] & (verdict.cells["joint_power"] < registered.POWER_FLOOR)
]
print(f"outcome: {verdict.outcome}")
for name, holds in verdict.conditions.items():
    print(f"  {'holds' if holds else 'fails'}: {name}")
if len(met):
    cells = ", ".join(f"{f} ({p})" for f, p in zip(met["feature"], met["phase"], strict=True))
    print(f"cells with an effect of {min_effect_cp:g} cp or more in every generation: {cells}")
if len(weak):
    cells = ", ".join(f"{f} ({p})" for f, p in zip(weak["feature"], weak["phase"], strict=True))
    print(f"eligible cells the sample could not have detected: {cells}")
print(
    "This compares the evaluation with game results, which is not the evaluation's error; "
    "the games were played by older nets; and only an SPRT of a net with new inputs says "
    "whether they make the engine stronger."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## In numbers

# COMMAND ----------

for gen in generations:
    numbers = effects_by_generation[gen].merge(verdict.cells, on=["feature", "phase"])
    display(numbers.assign(generation=gen).round(3))
display(shares.round(4))
