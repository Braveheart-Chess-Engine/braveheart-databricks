"""Which shards a run reads: a seeded sample per generation, or every shard."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence

#: Label sets this analysis reads. The relabelled sets live under selfplay/relabel/ and are
#: laid out differently; comparing them is future work.
LABEL_SETS = ("original",)

_GENERATION = re.compile(r"/gen=(\d+)/")
_POSITIONS = re.compile(r"^(?P<shard>.+)/positions/(?P<name>[^/]+)\.txt\.gz$")


def check_label_set(name: str) -> str:
    """The label set to read; anything but the originals is refused."""
    if name not in LABEL_SETS:
        raise ValueError(
            f"label set {name!r}: only the original labels are read for now "
            "(the relabelled sets are laid out differently)"
        )
    return name


def choose_shards(
    paths_by_generation: Mapping[int, Sequence[str]],
    per_generation: int | None,
    seed: int = 0,
) -> dict[int, list[str]]:
    """Up to `per_generation` positions shards per generation, chosen with `seed` and
    sorted; every shard when `per_generation` is None. A generation with no shards is an
    error, never a silent gap."""
    if per_generation is not None and per_generation < 1:
        raise ValueError("per_generation must be at least 1, or None for every shard")
    empty = sorted(gen for gen, paths in paths_by_generation.items() if not paths)
    if empty:
        raise ValueError(f"no shards for generation {', '.join(map(str, empty))}")
    picked: dict[int, list[str]] = {}
    for gen in sorted(paths_by_generation):
        paths = sorted(paths_by_generation[gen])
        if per_generation is None or per_generation >= len(paths):
            picked[gen] = paths
        else:
            rng = random.Random(f"{seed}:{gen}")
            picked[gen] = sorted(rng.sample(paths, per_generation))
    return picked


def games_path_for(positions_path: str) -> str:
    """The games file of the same shard as a positions file."""
    match = _POSITIONS.match(positions_path)
    if match is None:
        raise ValueError(f"not a positions shard: {positions_path!r}")
    return f"{match['shard']}/games/{match['name']}.jsonl.gz"


def generation_of(path: str) -> int:
    """The generation a shard's path is under (`.../gen=N/...`)."""
    match = _GENERATION.search(path)
    if match is None:
        raise ValueError(f"no gen=N in {path!r}")
    return int(match.group(1))
