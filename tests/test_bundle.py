"""The bundle's configuration, read as YAML: what deploys where, as whom, and how the job
runs. The offline schema check (CI) covers the rest of the CLI's rules."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = yaml.safe_load((ROOT / "databricks.yml").read_text())
JOB = yaml.safe_load((ROOT / "resources" / "selfplay_calibration.job.yml").read_text())[
    "resources"
]["jobs"]["selfplay_calibration"]
NOTEBOOK = (ROOT / "notebooks" / "selfplay_calibration.py").read_text()
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def test_the_bundle_and_its_targets() -> None:
    assert BUNDLE["bundle"]["name"] == "braveheart-databricks"
    assert BUNDLE["include"] == ["resources/*.yml"]
    dev, prod = BUNDLE["targets"]["dev"], BUNDLE["targets"]["prod"]
    assert dev["mode"] == "development"
    assert dev["default"] is True
    assert prod["mode"] == "production"
    assert prod["git"]["branch"] == "main"


def test_prod_runs_as_the_deployer_and_lets_the_workspace_users_view_and_run() -> None:
    # The design: users view and run what the deployer deploys. CAN_RUN, not
    # CAN_MANAGE: the deployed files stay read-only, so nobody changes what runs
    # as the deployer. The deployer manages its own deployment (the CLI's advice).
    prod = BUNDLE["targets"]["prod"]
    assert prod["run_as"] == {"service_principal_name": "${var.deployer_application_id}"}
    assert prod["permissions"] == [
        {"service_principal_name": "${var.deployer_application_id}", "level": "CAN_MANAGE"},
        {"group_name": "braveheart-databricks-workspace-users", "level": "CAN_RUN"},
    ]


def test_prod_deploys_to_one_fixed_folder_whoever_deploys() -> None:
    # Production mode wants a fixed root path, so only one copy exists; the deployer's
    # own folder, named by the variable, so no identifier is committed.
    assert BUNDLE["targets"]["prod"]["workspace"] == {
        "root_path": "/Workspace/Users/${var.deployer_application_id}"
        "/.bundle/${bundle.name}/${bundle.target}"
    }


def test_no_identifier_is_committed() -> None:
    # The repository is public: host, application ID and account ID are GitHub
    # environment variables, never files.
    for path in [ROOT / "databricks.yml", *sorted((ROOT / "resources").glob("*.yml"))]:
        text = path.read_text()
        assert not UUID.search(text), path
        assert "cloud.databricks.com" not in text, path


def test_the_job_runs_the_notebook_on_serverless_and_only_by_hand() -> None:
    (task,) = JOB["tasks"]
    path = task["notebook_task"]["notebook_path"]
    assert path == "../notebooks/selfplay_calibration.py"
    assert (ROOT / "resources" / path).resolve().is_file()
    assert not {"new_cluster", "existing_cluster_id", "job_cluster_key"} & set(task)
    assert not {"job_clusters", "schedule", "trigger", "continuous"} & set(JOB)
    assert JOB["max_concurrent_runs"] == 1


def test_the_job_parameters_and_their_defaults() -> None:
    assert {p["name"]: p["default"] for p in JOB["parameters"]} == {
        "shards_per_generation": "24",
        "generations": "0,1,2,3,4,5,6",
        "label_set": "original",
    }


def test_the_notebook_takes_the_job_parameters_with_the_same_defaults() -> None:
    assert NOTEBOOK.startswith("# Databricks notebook source\n")
    widgets = dict(re.findall(r'dbutils\.widgets\.text\("([a-z_]+)", "([^"]*)"', NOTEBOOK))
    assert widgets == {p["name"]: p["default"] for p in JOB["parameters"]}


def test_the_notebook_installs_the_plotly_the_tests_use() -> None:
    pinned = re.search(r'"plotly==([0-9.]+)"', (ROOT / "pyproject.toml").read_text())
    assert pinned
    assert re.search(
        rf"^# MAGIC %pip install --quiet plotly=={re.escape(pinned.group(1))}$", NOTEBOOK, re.M
    )
