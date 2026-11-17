# Contributing

This repository holds two things:

- [`jaws.yaml`](jaws.yaml), what this Databricks workspace
  project asks for. [jaws](https://github.com/jwallace145/jaws)
  provisions the workspace from it.
- A Databricks asset bundle, [`databricks.yml`](databricks.yml): analysis notebooks (`notebooks/`), the
  jobs that run them (`resources/`) and their tested Python package (`src/selfplay_analysis/`).

## Setup

```sh
npm ci                                                          # commitlint
uv sync                                                         # Python, for the bundle's package
pre-commit install --hook-type commit-msg --hook-type pre-commit
```

Before a pull request, run the tests and the linters:

```sh
uv run pytest                    # the Spark tests need a JDK 17 or 21; they skip without one, never in CI
uv run ruff check . && uv run ruff format --check .
```

The hooks are a convenience. CI is the authoritative check.

## Commit messages

Every commit is a **single-line** [Conventional Commit](https://www.conventionalcommits.org/)
of at most 100 characters. Details go in the pull request description, never in a commit body.

- Format: `type(scope)?: subject`, with an imperative, lowercase subject and no trailing period.
- The only thing allowed after the header is a blank line followed by
  `Co-authored-by: Name <email>` trailers.
- Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `ci`, `build`, `perf`, `style`, `test`, `revert`.
- A breaking change is marked with `!`, as in `feat!: drop a compute policy`.
- The type drives the next version: `feat` bumps the minor version, `fix` the patch,
  and `!` the major (the minor while the version is below 1.0).

| Message | Valid |
| --- | --- |
| `feat: add a workspace user` | Yes |
| `fix: correct a policy name` + blank line + `Co-authored-by: Claude <noreply@anthropic.com>` | Yes |
| `feat: add thing` + blank line + `This adds the thing because…` | No, body text |
| `Added thing` | No, not conventional |

CI lints every commit in the pull request, and the pull request title separately.

## Pull requests

Every change reaches `main` through a pull request. Nobody pushes to `main`.

- The title is a valid Conventional Commit: squash merges use it as the commit message.
- Open the pull request with a typed template and fill in every section:

  ```sh
  gh pr create --template <feat|fix|docs|chore|refactor|ci>.md
  ```

- Pull requests merge by squash or rebase only. There are no merge commits.

## How a change reaches the workspace

**The bundle** deploys when its change merges: the **Deploy** workflow signs in
as the project's deployer (GitHub OIDC, no secret) and runs
`databricks bundle deploy -t prod`. Nobody deploys `prod` by hand; deploy the
`dev` target from your laptop to try a change first (README, *Developing*).

**The project file** is applied by the infra repository, which reads it from
this repository's `main`:

1. Merge the change to `main`.
2. The next pull request in jaws shows it in its **Databricks plan** comment.
3. The owner runs the **Databricks apply** workflow in jaws for this project.

The network range, cost center, workspace admins and the compute policies the
project may use are set centrally, in jaws's `config.yaml`.
The fields of the project file are documented in
[docs/databricks.md](https://github.com/jwallace145/jaws/blob/main/docs/databricks.md#the-project-file).

## Releases

[Release Please](https://github.com/googleapis/release-please) keeps a release
pull request open with the next version and the changelog. Merging it tags the
version. It owns `CHANGELOG.md`; nobody edits that file by hand.
