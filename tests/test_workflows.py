"""The workflows, read as YAML: pull requests check offline with no credential, and only
main deploys, as the deployer, never running a job."""

from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _load(name: str) -> dict[str, Any]:
    workflow = yaml.safe_load((WORKFLOWS / name).read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))  # YAML 1.1 reads `on` as true
    return workflow


DEPLOY = _load("deploy.yml")
CI = _load("ci.yml")
(DEPLOY_JOB,) = DEPLOY["jobs"].values()


def _runs(job: dict[str, Any]) -> list[str]:
    return [step["run"].strip() for step in job["steps"] if "run" in step]


def test_deploy_runs_on_pushes_to_main_and_by_hand() -> None:
    assert DEPLOY["on"] == {"push": {"branches": ["main"]}, "workflow_dispatch": None}
    assert DEPLOY_JOB["if"] == "github.ref == 'refs/heads/main'"


def test_deploy_runs_in_prod_with_only_an_id_token() -> None:
    assert DEPLOY["permissions"] == {}
    assert DEPLOY_JOB["environment"] == "prod"
    assert DEPLOY_JOB["permissions"] == {"id-token": "write", "contents": "read"}
    assert DEPLOY["concurrency"]["cancel-in-progress"] is False


def test_deploy_signs_in_as_the_deployer_from_environment_variables() -> None:
    assert DEPLOY_JOB["env"] == {
        "DATABRICKS_AUTH_TYPE": "github-oidc",
        "DATABRICKS_HOST": "${{ vars.DATABRICKS_HOST }}",
        "DATABRICKS_CLIENT_ID": "${{ vars.DATABRICKS_CLIENT_ID }}",
        "DATABRICKS_TOKEN_AUDIENCE": "${{ vars.DATABRICKS_TOKEN_AUDIENCE }}",
        "BUNDLE_VAR_deployer_application_id": "${{ vars.DATABRICKS_CLIENT_ID }}",
    }


def test_deploy_checks_who_it_is_then_validates_and_deploys_never_running_a_job() -> None:
    assert _runs(DEPLOY_JOB) == [
        "databricks current-user me",
        "databricks bundle validate -t prod",
        "databricks bundle deploy -t prod",
    ]


def test_the_cli_is_pinned_to_a_release_everywhere() -> None:
    uses = [
        step["uses"]
        for workflow in (DEPLOY, CI)
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("databricks/setup-cli@")
    ]
    assert uses and set(uses) == {"databricks/setup-cli@v1.26.0"}


def test_pull_request_checks_hold_no_credential() -> None:
    assert "pull_request" in CI["on"]
    for name in ("python", "bundle-schema"):
        job = CI["jobs"][name]
        assert job["permissions"] == {"contents": "read"}
        text = yaml.safe_dump(job)
        assert "vars." not in text and "secrets." not in text


def test_ci_runs_the_spark_tests_for_real() -> None:
    job = CI["jobs"]["python"]
    assert job["env"]["SPARK_TESTS"] == "required"
    assert any(step.get("uses", "").startswith("actions/setup-java@") for step in job["steps"])
    assert "uv run pytest" in _runs(job)


def test_ci_checks_the_bundle_against_the_clis_schema_offline() -> None:
    runs = "\n".join(_runs(CI["jobs"]["bundle-schema"]))
    assert "databricks bundle schema" in runs and "check-jsonschema" in runs
    assert "bundle validate" not in runs  # it needs a workspace login
