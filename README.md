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
repository only says what the project asks for: its compute policies and its
workspace users. The network, cost center, admins and guardrails are central.
See [docs/databricks.md](https://github.com/jwallace145/jwall-engineering-infra/blob/main/docs/databricks.md).

The NNUE datasets stay owned by Braveheart (its `nnue` dataset in its own
`jwall-engineering-sites.yaml`); this workspace is to be granted read-only
access to them, declared on Braveheart's side.

## Contributing

Changes go through pull requests with one-line Conventional Commits; see
[CONTRIBUTING.md](CONTRIBUTING.md). The rules for coding agents are in
[AGENTS.md](AGENTS.md). Licensed under the [MIT License](LICENSE).
