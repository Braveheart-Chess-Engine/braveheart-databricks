"""The structural analysis's job and notebook, read as text: how the job runs, that its
defaults are the registered values, and that the notebook checks the net before it reads a
position and never states a verdict it did not compute."""

import re
from pathlib import Path

import yaml

from selfplay_analysis import registered

ROOT = Path(__file__).resolve().parent.parent
JOB = yaml.safe_load((ROOT / "resources" / "nnue_structural_analysis.job.yml").read_text())[
    "resources"
]["jobs"]["nnue_structural_analysis"]
NOTEBOOK = (ROOT / "notebooks" / "nnue_structural_analysis.py").read_text()
CELLS = NOTEBOOK.split("# COMMAND ----------")
DEFAULTS = {p["name"]: p["default"] for p in JOB["parameters"]}


def test_the_job_runs_the_notebook_on_serverless_and_only_by_hand() -> None:
    assert JOB["name"] == "nnue-structural-analysis"
    (task,) = JOB["tasks"]
    path = task["notebook_task"]["notebook_path"]
    assert path == "../notebooks/nnue_structural_analysis.py"
    assert (ROOT / "resources" / path).resolve().is_file()
    assert not {"new_cluster", "existing_cluster_id", "job_cluster_key"} & set(task)
    assert not {"job_clusters", "schedule", "trigger", "continuous"} & set(JOB)
    assert JOB["max_concurrent_runs"] == 1
    assert JOB["tags"] == {"project": "braveheart-databricks"}


def test_the_job_parameters_and_their_defaults() -> None:
    assert DEFAULTS == {
        "generations": "5,6",
        "shards_per_generation": "1200",
        "sampling_seed": "0",
        "min_effect_cp": "15",
        "max_positions_per_game": "12",
        "exclude_draw_adjudicated": "false",
        "net_path": "",
    }


def test_the_defaults_are_the_registered_values() -> None:
    assert float(DEFAULTS["min_effect_cp"]) == registered.MIN_EFFECT_CP
    assert int(DEFAULTS["shards_per_generation"]) == registered.SHARDS_PER_GENERATION
    assert int(DEFAULTS["max_positions_per_game"]) == registered.MAX_POSITIONS_PER_GAME
    assert tuple(int(g) for g in DEFAULTS["generations"].split(",")) == registered.GENERATIONS


def test_the_notebook_takes_the_job_parameters_with_the_same_defaults() -> None:
    assert NOTEBOOK.startswith("# Databricks notebook source\n")
    widgets = dict(re.findall(r'dbutils\.widgets\.text\("([a-z_]+)", "([^"]*)"', NOTEBOOK))
    assert widgets == DEFAULTS


def test_the_notebook_installs_the_plotly_the_tests_use() -> None:
    pinned = re.search(r'"plotly==([0-9.]+)"', (ROOT / "pyproject.toml").read_text())
    assert pinned
    assert re.search(
        rf"^# MAGIC %pip install --quiet plotly=={re.escape(pinned.group(1))}$", NOTEBOOK, re.M
    )


def test_the_net_is_checked_against_the_engine_before_a_position_is_read() -> None:
    assert "sha256=nnue.NET_SHA256" in NOTEBOOK
    assert NOTEBOOK.index("nnue.check_golden(") < NOTEBOOK.index("read_shards(spark")


def test_the_notebook_reads_only_the_read_only_volume() -> None:
    assert 'VOLUME = "/Volumes/braveheart_selfplay/positions/files/selfplay/v1"' in NOTEBOOK
    assert not re.search(r"\.write\b|saveAsTable|dbutils\.fs\.(put|rm|cp|mv)", NOTEBOOK)


def test_the_verdict_is_computed_never_typed() -> None:
    # The first cell describes the rule and may name its outcomes. No later Markdown does:
    # what the run found is said by the code, from the verdict.
    assert "decide(" in NOTEBOOK and "verdict_banner(" in NOTEBOOK
    for cell in CELLS[1:]:
        prose = "\n".join(line for line in cell.splitlines() if line.startswith("# MAGIC"))
        assert not re.search(r"\b(GO|NO-GO|INCONCLUSIVE)\b", prose), prose


def test_every_generation_is_analysed_on_its_own_with_the_registered_rule() -> None:
    assert "decide(effects_by_generation, shuffled_by_generation, min_effect_cp)" in NOTEBOOK
    assert "registered.MIN_SHARE" in NOTEBOOK


def test_findings_are_stated_only_when_the_rule_reaches_them() -> None:
    # The cells that met the size condition are named on GO alone, and a run that decides
    # nothing says so on the charts and tables that would otherwise read as findings.
    assert 'if verdict.outcome == "GO":' in NOTEBOOK
    assert NOTEBOOK.count("caveat=caveat") == 2  # both forests
    assert "interpretable=" in NOTEBOOK  # the tables


def test_a_run_with_other_parameters_is_not_called_the_registered_result() -> None:
    for name in ("MIN_EFFECT_CP", "MAX_POSITIONS_PER_GAME", "GENERATIONS"):
        assert f"registered.{name}" in NOTEBOOK
    assert "as_registered=as_registered" in NOTEBOOK
