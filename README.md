<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Braveheart-Chess/braveheart-branding/main/banner-dark.png">
    <img src="https://raw.githubusercontent.com/Braveheart-Chess/braveheart-branding/main/banner-light.png" alt="Braveheart" width="640">
  </picture><br>
  <sub>Logo © the artist, used under licence. All rights reserved; not covered by this repository's licence (<a href="https://github.com/Braveheart-Chess/braveheart-branding#licence">details</a>).</sub>
</p>

# braveheart-databricks

The Databricks workspace for [Braveheart](https://github.com/Braveheart-Chess/braveheart),
the UCI chess engine: a governed place to explore and analyse its NNUE training
data with Unity Catalog, central compute policies, audit logs and tags on every
resource.

The workspace is provisioned by
[jaws](https://github.com/jwallace145/jaws)
from [`jaws.yaml`](jaws.yaml). This
repository only says what the project asks for: its compute policies, its
workspace users, the datasets it reads and a deploy identity for its CI
(`deployer: true`: a service principal GitHub Actions signs in as, with no
secret, to deploy this repository's analyses). The network, cost center, admins
and guardrails are central.
See [docs/databricks.md](https://github.com/jwallace145/jaws/blob/main/docs/databricks.md).

## Datasets it reads

The workspace reads Braveheart's self-play positions, **read-only**. Access is a
handshake on a dataset the platform has onboarded, and needs all three:

- jaws onboards the dataset
  (`datasets/braveheart-selfplay/positions.yaml`): in Unity Catalog it is the
  schema `positions` in braveheart-selfplay's catalog, `braveheart_selfplay`,
  holding one external volume, `files`;
- [braveheart-selfplay](https://github.com/Braveheart-Chess/braveheart-selfplay),
  which owns the positions, lists this project under the dataset's `consumers`;
- this repository lists the dataset under the workspace's `reads`, with its
  purpose.

Either side revokes access alone by removing its half. With both halves, the
catalog `braveheart_selfplay` appears in this workspace's Catalog Explorer,
bound read-only, next to `braveheart_databricks`. Nothing here can write to the
data: the workspace users group holds `USE_CATALOG`, `USE_SCHEMA` and
`READ_VOLUME` only, and the volume's storage is read-only. Other workspaces
see only what is shared with them
([how it works](https://github.com/jwallace145/jaws/blob/main/docs/databricks.md#datasets-and-access)).

### Reading the positions

Once access is live, the data is under
`/Volumes/braveheart_selfplay/positions/files/selfplay/`:
`v1/gen=<N>/run=<run>/task=<t>/` holds gzipped `FEN | score | result` records
in `positions/`, the full games in `games/` and one manifest per shard in
`manifests/`, and `relabel/<label set>/gen=<N>/` holds relabelled scores
([layout](https://github.com/Braveheart-Chess/braveheart-selfplay/blob/main/docs/datagen.md#layout-in-s3),
[what each generation holds](https://github.com/Braveheart-Chess/braveheart-selfplay/blob/main/docs/datasets.md)).
A generation is hundreds of millions of records: list a run's `manifests/`
first, and start from one run.

```python
from pyspark.sql import functions as F

base = "/Volumes/braveheart_selfplay/positions/files/selfplay/v1"
display(dbutils.fs.ls(f"{base}/gen=6/"))

records = spark.read.text(f"{base}/gen=6/run=<run>/task=t0/positions/*.txt.gz")
fields = F.split("value", r" \| ")
positions = records.select(
    fields[0].alias("fen"),
    fields[1].cast("int").alias("score"),  # centipawns, white's point of view
    fields[2].cast("double").alias("result"),  # 1.0, 0.5 or 0.0 for white
)
```

## Analyses

This repository is also a
[Databricks asset bundle](https://docs.databricks.com/aws/en/dev-tools/bundles/)
([`databricks.yml`](databricks.yml)): analysis notebooks and the jobs that run
them, version-controlled here and deployed by CI.

| What | Where |
| --- | --- |
| The self-play calibration report: do scores predict results, generation by generation? | [`notebooks/selfplay_calibration.py`](notebooks/selfplay_calibration.py), job `selfplay-calibration` |
| The NNUE structural analysis: is the net's raw evaluation wrong by pawn structure by enough to justify new inputs? GO, NO-GO or INCONCLUSIVE, by a rule registered before any result | [`notebooks/nnue_structural_analysis.py`](notebooks/nnue_structural_analysis.py), job `nnue-structural-analysis`, [the research note](docs/nnue-structural-research.md) |
| Their tested logic: parsing, phase, the scale fit, pawn structure, the net's evaluation in NumPy, the models, the rule, the charts | [`src/selfplay_analysis/`](src/selfplay_analysis) |

### Running the report

In the workspace, open **Jobs & Pipelines**, then **selfplay-calibration**:

- **Run now** samples 24 shards a generation, about 4 million positions, in a
  few minutes;
- **Run now with different parameters** takes `shards_per_generation` (`all`
  reads every shard, about 3.2 billion positions), `generations` and
  `label_set` (only `original` for now).

The run's notebook output is the report: interactive charts of calibration, the
best-fitting scale, where the fit breaks by phase, and each generation's
results, scores, phases and games. The deployed notebook also runs
interactively, with the same parameters as widgets.

### Running the structural analysis

**nnue-structural-analysis** runs the same way. **Run now** samples 672 shards a
generation of generations 5 and 6 (about 33.6 million positions a pass, 8.6 million of
them in the models), evaluates each position with the engine's current net and
measures, per pawn-structure feature and phase, how far the evaluation is from the
results. Its last chart is the verdict. What the rule is, why the sample is that
size, what each parameter does and what the report cannot say are in
[the research note](docs/nnue-structural-research.md).

The report downloads the net file from the engine's public repository and checks its
SHA-256. If the workspace cannot reach it, upload the file and give its path as
`net_path`.

### Developing

```sh
uv sync                                          # Python 3.12, PySpark, Plotly, pytest, ruff
uv run pytest                                    # the Spark tests need a JDK 17 or 21; they skip without one
                                                 # the NNUE port's test needs the net: tests/fixtures/<net id>.nnue or BRAVEHEART_NET
uv run ruff check . && uv run ruff format --check .
```

To try a change in the workspace before merging, deploy a private copy under
your own login:

```sh
databricks auth login --host <workspace URL>
databricks bundle deploy                         # target dev: names prefixed [dev <you>]
databricks bundle run selfplay_calibration
```

### How deploys work

A merge to `main` runs **Deploy** ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)):
it signs in as the project's deployer, a workspace service principal that
jaws creates because the project file sets `deployer: true`,
through GitHub OIDC with no secret, and deploys the bundle's `prod` target.
Jobs run as the deployer, which reads exactly what the workspace's users may.
The GitHub environment `prod` (deployable from `main` only) and its variables
(`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_TOKEN_AUDIENCE`) are
the owner's.

## Contributing

Changes go through pull requests with one-line Conventional Commits; see
[CONTRIBUTING.md](CONTRIBUTING.md). The rules for coding agents are in
[AGENTS.md](AGENTS.md). Licensed under the [MIT License](LICENSE).
