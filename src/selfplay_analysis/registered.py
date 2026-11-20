"""The values the structural analysis was registered with, before any result was seen
(`docs/nnue-structural-research.md`). Changing one after a run changes the question."""

from __future__ import annotations

#: The generations analysed, each on its own: the rule asks them to agree.
GENERATIONS = (5, 6)
#: The smallest structure effect that matters, in centipawns of evaluation per unit of a
#: feature's white-minus-black difference. An estimate of at least this in every generation
#: is the evidence for an effect; every interval inside it is the evidence against one.
MIN_EFFECT_CP = 15.0
#: A feature in a phase is eligible when the sides differ in at least this share of the
#: phase's positions, in every generation.
MIN_SHARE = 0.05
#: Worker groups are split into this many folds for the cross-fitted predictions.
FOLDS = 5
#: Game resamples behind each interval of a gain.
BOOTSTRAP = 1000
#: A game contributes at most this many positions, evenly spaced: its positions share one
#: result, so more of them add rows faster than information (12 instead of 24 raised the
#: median standard error by 3.7%).
MAX_POSITIONS_PER_GAME = 12
#: Evaluations are clipped to this many centipawns either way before fitting.
EVAL_CLIP = 2000
#: The ridge penalty of the cross-fitted models (the effects are fitted without one).
RIDGE = 1.0
#: Shards sampled a generation: enough that, were no effect real, every eligible cell's
#: interval would lie inside the threshold in both generations about nine times in ten,
#: from standard errors measured before any effect was looked at.
SHARDS_PER_GENERATION = 1200
#: The most rows collected to the driver for the model; beyond it, whole games are sampled.
#: The default sample is about 8.2 million.
MAX_MODEL_ROWS = 12_000_000
