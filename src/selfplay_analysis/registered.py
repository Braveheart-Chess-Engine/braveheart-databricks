"""The values the structural analysis was registered with, before any result was seen
(`docs/nnue-structural-research.md`). Changing one after a run changes the question."""

from __future__ import annotations

#: The smallest structure effect that matters, in centipawns of evaluation per unit of a
#: feature's white-minus-black difference.
MIN_EFFECT_CP = 15.0
#: A feature in a phase is eligible when the sides differ in at least this share of the
#: phase's positions, in every generation.
MIN_SHARE = 0.05
#: Every eligible cell must be detectable in every generation at once with this power.
POWER_FLOOR = 0.80
#: Worker groups are split into this many folds for the cross-fitted predictions.
FOLDS = 5
#: Game resamples behind each interval of a gain.
BOOTSTRAP = 1000
#: A game contributes at most this many positions, evenly spaced: its positions share one
#: result, so more of them add rows faster than information.
MAX_POSITIONS_PER_GAME = 24
#: Evaluations are clipped to this many centipawns either way before fitting.
EVAL_CLIP = 2000
#: The ridge penalty of the cross-fitted models (the effects are fitted without one).
RIDGE = 1.0
#: Shards sampled a generation: what the power condition needs for its least-powered
#: eligible cell (passed pawns in the opening, about 643), from standard errors measured
#: before any effect was looked at.
SHARDS_PER_GENERATION = 672
#: The most rows collected to the driver for the model; beyond it, whole games are sampled.
#: The default sample is about 8.6 million.
MAX_MODEL_ROWS = 12_000_000
