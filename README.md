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
[jwall-engineering-infra](https://github.com/jwallace145/jwall-engineering-infra)
from [`jwall-engineering-sites.yaml`](jwall-engineering-sites.yaml). This
repository only says what the project asks for: its compute policies, its
workspace users and the datasets it reads. The network, cost center, admins and
guardrails are central.
See [docs/databricks.md](https://github.com/jwallace145/jwall-engineering-infra/blob/main/docs/databricks.md).

## Datasets it reads

The workspace reads Braveheart's self-play positions, **read-only**. Access is a
handshake on a dataset the platform has onboarded, and needs all three:

- jwall-engineering-infra onboards the dataset
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
([how it works](https://github.com/jwallace145/jwall-engineering-infra/blob/main/docs/databricks.md#datasets-and-access)).

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

## Contributing

Changes go through pull requests with one-line Conventional Commits; see
[CONTRIBUTING.md](CONTRIBUTING.md). The rules for coding agents are in
[AGENTS.md](AGENTS.md). Licensed under the [MIT License](LICENSE).
