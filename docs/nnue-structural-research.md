# Does the net misjudge pawn structure?

A research note for one decision: whether Braveheart's current net,
`net8x160-l050-r43390801430`, is wrong about pawn structure by enough to justify
training a net with explicit pawn-relationship inputs (sparse pawn-pair features).

The report that answers it is the job `nnue-structural-analysis`
([notebook](../notebooks/nnue_structural_analysis.py)). This note is its
registration: the rule below was written, and merged, before the report had run on
real data. The note holds no result. The report computes the verdict from the rule
each time it runs.

## The question

The net has 768 piece-square inputs per perspective, four king buckets, eight output
buckets and 512 hidden neurons. It can represent an isolated or a passed pawn, but
only by combining squares in its hidden layer. Braveheart's pawn-structure correction
history gained Elo by correcting the evaluation per pawn structure during a search,
which suggests the net leaves structure-shaped error behind.

The question is whether that error shows in the net's **raw static evaluation** of
self-play positions, as a pattern that pawn-structure features explain and the
evaluation does not: among positions the net scores the same, does the side with,
say, the extra passed pawn go on to score better than the evaluation says?

This is a filter, not proof. Only training a second net and testing it by SPRT shows
that new inputs make the engine stronger. The report decides whether that experiment
is worth its cost. "No" is a successful outcome.

## The rule

A **cell** is one structure feature in one phase (opening, middlegame, endgame). A cell
is **eligible** when the feature differs between the sides in at least
5% of that phase's positions, in every generation analysed.

An **effect** is measured in centipawns of evaluation: how far the evaluation would
have to move, per unit of the feature's white-minus-black difference, to predict the
results it missed. `MIN_EFFECT_CP` = 15 cp is the smallest effect that matters. It is a
judgement, not a derivation, fixed before any result and not to be changed after a
run. For scale: a 5,000-node score moves 17 to 30 cp from search state alone
([relabel.md](https://github.com/Braveheart-Chess/braveheart-selfplay/blob/main/docs/relabel.md)).

The report analyses generations 5 and 6 separately. They are independent samples:
different engines played them, in different runs, on different workers. The three
conditions:

1. **Size.** Some eligible cell has an effect of at least `MIN_EFFECT_CP`, with the
   same sign in every generation, and each generation's 95% interval excludes zero.
2. **Control.** The same analysis, with the features shuffled so they no longer
   belong to their positions, finds no such cell.
3. **Power.** Every eligible cell has at least 80% power to be detected in every
   generation at once (the product of the generations' powers) if its effect were
   exactly `MIN_EFFECT_CP`.

The three outcomes:

| Outcome | When | What follows |
|:--|:--|:--|
| **GO** | All three conditions hold | Design the pawn-pair net comparison ([below](#if-the-outcome-is-go)) |
| **NO-GO** | Conditions 2 and 3 hold, and condition 1 fails | The data could have shown an effect of the size that matters and did not. Record it in the engine's research log |
| **INCONCLUSIVE** | Anything else, or fewer than two generations | With condition 3 failing: raise `shards_per_generation` and run again. With condition 2 failing: the method found an effect where none can be, and is suspect. Nothing is read from either run |

**What the rule does not ask for.** The report also measures how much the structure
features improve the prediction of results on held-out workers (the *gain*), with the
shuffled features beside it. It is shown, and is not a condition. Predicting better
is a much weaker test than measuring a coefficient. The 21 structure coefficients are
fitted together, and each costs a little prediction whether or not it is real. In
simulation the gain's z-score followed (L − k) / √(4L + 2k), where L is the sum of the
cells' squared z-scores and k = 21 is the number of coefficients: 3.2 at L = 91 and
8.8 at L = 292. One cell at the edge of the power condition has L of about 10, less
than k, so its expected gain is below zero. A first draft of this rule made the gain
a condition and let NO-GO follow from its failing, which would have called a real
15 cp effect NO-GO. It was replaced before the report had read a real position.

The registered values live in
[`registered.py`](../src/selfplay_analysis/registered.py), and a test holds this note
and the job's defaults to them.

## What the data is

Self-play positions from
[braveheart-selfplay](https://github.com/Braveheart-Chess/braveheart-selfplay), read
through the volume `braveheart_selfplay.positions.files`, read-only. Each record is a
FEN, the search's score and the game's result, both from white's point of view. Each
shard also has its games in full: the start position, the moves, a score per move,
the result and how the game ended.

Before anything was built, six questions were answered on a sample drawn as the
report draws its own (seed 0, 96 shards a generation, generations 5 and 6), with 12
shards a generation read in full. Every count below is from that sample.

| Question | Answer |
|:--|:--|
| Can each record be given its game? | Yes. A replay of every game with a chess library placed all 601,283 records of 24 shards in their 6,647 games, and the method the report uses (the FEN's move counter, the result and the score, with no chess logic) placed every record identically. No game was left without a record |
| How do games end? | Of 53,137 games in the 192 sampled shards: win adjudication 52.4%, repetition 28.3%, draw adjudication 8.7%, insufficient material 5.4%, fifty moves 4.2%, checkmate 0.3%, stalemate 0.3%, the ply limit 0.5%. Of the decisive games read in full, 3,435 of 3,460 were win-adjudicated |
| How many independent groups? | 96 shards cover 77 of generation 5's 192 worker groups and 78 of generation 6's 320: 25,749 and 27,388 games, 2,405,970 and 2,405,770 positions, 47% of games drawn in both |
| Is the net's raw evaluation available? | Yes: `Board::evaluate()` in the engine's search crate at `v0.54.0`, whose embedded net is `net8x160-l050-r43390801430`. The net file is in the engine's public repository |
| Can new labels be added to the dataset for a sample of shards? | Not sensibly. The relabel workflow works by generation and worker, it is the only writer to a bucket that keeps everything forever, and a label set for 192 shards would sit there incomplete |
| How many shards does the power condition need? | About 643 a generation; the default is 672. Standard errors were measured on the 12 shards a generation read in full, with the effects themselves left unprinted, and they fall with the square root of the games. The least-powered eligible cell is passed pawns in the opening (the sides differ in 13% of opening positions; a standard error of 34 cp at 12 shards). The next need 168 to 193 shards. Keeping 16 positions a game instead of 24 would raise the median standard error by 0.8%, and 12 by 3.7% |

Three of those answers shaped the method. The sample is 672 shards a generation
because the power condition needs it. Nearly every decisive game ends by win
adjudication (1,200 cp for four plies), so the result target uses every game: games
that ended "naturally" are almost all draws. And the evaluation is computed inside
the report rather than stored as a label set.

## Method

**Sample.** `shards_per_generation` shards a generation, chosen with `sampling_seed`
by the same function as the calibration report. A shard is whole games from one
worker. Every sampled shard's line count is checked against its manifest.

**Games.** Each record is given its game and its ply within it from the shard's games
file. A shard whose records do not match its games is counted and left out, never
guessed at.

**Thinning.** A game's positions share one result, so they are far from independent.
Each game contributes at most 24 positions, evenly spaced through it.

**Features.** For each side, from the FEN's board field alone:

| Feature | Definition |
|:--|:--|
| `pawns` | Pawns on the board (part of the baseline: it is material, not structure) |
| `isolated` | Pawns with no friendly pawn on a neighbouring file |
| `doubled` | Pawns beyond the first on a file |
| `islands` | Runs of neighbouring files that hold a pawn |
| `connected` | Pawns defended by a friendly pawn, or with a friendly pawn beside them on the same rank |
| `passed` | Pawns with no enemy pawn ahead on their own or a neighbouring file |
| `protected_passed` | Passed pawns defended by a friendly pawn |
| `passed_rank` | The most advanced passed pawn's rank, counted from its own side (2 to 7); 0 with none |

Each enters the model as white's count minus black's.

**Evaluation.** The net's raw static evaluation of each position, from white's point
of view: no search and no correction history. It is computed by a NumPy port of the
engine's inference
([`nnue.py`](../src/selfplay_analysis/nnue.py)) from the net file at the engine's
`v0.54.0`, whose SHA-256 the report checks. Before reading any position, the report
evaluates a set of positions the engine itself evaluated and stops unless every one
matches exactly.

The golden positions
([`net8x160-golden.tsv`](../src/selfplay_analysis/data/net8x160-golden.tsv)) are 480
positions from the 24 shards read in full, drawn with seed 0: 30 for each of the
eight output buckets and each side to move, with both kings on either half of the
board. The engine evaluated them through this program, built against its `v0.54.0`
crates:

```rust
use std::io::{self, BufRead, Write};

use braveheart_core::position::Position;
use braveheart_core::types::Color;
use braveheart_search::board::Board;
use braveheart_search::nnue::default_network;

fn main() {
    let net = default_network();
    let mut out = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let fen = line.expect("a line");
        let pos = Position::from_fen(&fen).expect("a FEN");
        let white_to_move = pos.side_to_move() == Color::White;
        let mut board = Board::from_position(pos);
        board.set_network(Some(net.clone()));
        let eval = board.evaluate();
        writeln!(out, "{fen}\t{}", if white_to_move { eval } else { -eval }).expect("stdout");
    }
}
```

Run once over every position of those 24 shards, the port and the engine agreed on
all 601,283.

**Model.** Per generation, a logistic model of the game's result (a draw counts half,
so the log loss is the cross-entropy against 0, ½ or 1):

- **Baseline:** the evaluation, the difference in non-pawn material and the difference
  in pawns, each with its own coefficient per phase; the side to move; an intercept.
- **With structure:** the baseline plus the seven structure differences, each per
  phase.

**Effects** come from one fit of the structure model on all of a generation's games,
with standard errors clustered by game. An effect is 100 times the structure
coefficient over the phase's evaluation coefficient (the evaluation enters in units of
100 cp), with its interval by the delta method. A **positive** effect means the side
with more of the feature scores better than the evaluation says: the net undervalues
the feature. **Power** is the chance such an interval would exclude zero were the true
effect `MIN_EFFECT_CP`, from the same standard error.

**Control.** The structure differences are permuted across positions within cells of
phase and evaluation (100 cp wide), a position's seven differences moving together.
That keeps their distribution and breaks their link to the result. The effects are
then measured again, exactly as before, and the size condition must not hold.

**Gain** (shown, not a condition) is the drop in log loss from adding the structure
terms, on predictions each position gets from a model fitted without its worker group
(five folds of worker groups, so no worker is on both sides). Its interval comes from
resampling games 1,000 times. The shuffled features' gain is shown beside it.

**The estimator's own tests.** Two properties are automated tests rather than report
sections: a planted effect of 15 cp is recovered as 15 cp, and when nothing is planted
the clustered intervals exclude zero about 5% of the time (4.1% over 4,200 simulated
intervals of 3,000 games each).

**Sensitivity.** `exclude_draw_adjudicated` reruns everything without the games that
ended by draw adjudication (from ply 80, twelve plies within 10 cp), the one ending
that the engine's own evaluation decides and could decide wrongly.

## Limits

- **Evaluation against result is not evaluation error.** A game's result is one noisy
  outcome of imperfect play, not the position's value.
- **The games were played by older nets.** Generation 5 was played by net6k4 and
  generation 6 by net7rl012. If those engines misplayed a structure, the results
  reward it less than a stronger engine's would.
- **The net trained on these positions.** net8x160's training set is generations 0 to
  6 with their results at λ = 0.5
  ([datasets.md](https://github.com/Braveheart-Chess/braveheart-selfplay/blob/main/docs/datasets.md#which-labels-each-net-trained-on)).
  An effect here is the net failing to fit a pattern in its own targets, which is
  what a new input could fix, but it is measured mostly in-sample.
- **Quiet positions only.** Datagen keeps a position when the side to move is not in
  check, the best move is not a capture or promotion and the score is not a mate
  score, after eight or ten random plies. Nothing here speaks for tactical positions.
- **Adjudication.** Win adjudication is treated as a reliable result. Draw
  adjudication is the suspect ending, and the sensitivity run drops it.
- **The correction history already acts on this.** During a search the engine
  corrects the evaluation per pawn structure. A net that learned the same thing would
  compete with that table, so an SPRT may show less than an effect here suggests.
- **Seven hand-written features are not pawn-pair inputs.** An effect in `passed` says
  where the net is wrong, not that a particular input set will fix it.

## Running it

In the workspace, open **Jobs & Pipelines**, then **nnue-structural-analysis**, and
**Run now**. The parameters:

| Parameter | Default | Meaning |
|:--|:--|:--|
| `generations` | `5,6` | The generations analysed, each on its own. Fewer than two is INCONCLUSIVE |
| `shards_per_generation` | `672` | Shards sampled a generation. Fewer may leave the power condition unmet |
| `sampling_seed` | `0` | The sample's seed |
| `min_effect_cp` | `15` | `MIN_EFFECT_CP`. Registered: not to be changed after a run |
| `max_positions_per_game` | `24` | The thinning cap |
| `exclude_draw_adjudicated` | `false` | The sensitivity run |
| `net_path` | empty | A copy of the net file in the workspace. Empty downloads it from the engine's repository |

The run reads its shards twice, each time through one Python pass per shard, and
fits a handful of small models on the driver. It is a serverless notebook task, as
the calibration report is. One core turns a shard of about 25,000 positions into rows
in under a second, evaluation included: 26,400 to 27,500 positions a second on three
of the sampled shards, on an Apple M-series core. The default sample is 1,344 shards:
about 33.6 million positions a pass, of which about 8.6 million (24 a game) reach the
models.

## Not done here

- **The search's view.** Whether net8x160's 5,000-node score disagrees with its own
  static evaluation by structure needs a net8x160 label set, which does not exist.
- **Earlier nets.** How score-to-result calibration by structure moved from
  generation 0 to 6 needs the relabelled sets, which this package does not read yet.
- **A deeper reference.** A sample of positions searched far deeper would be a
  stronger target than a game result. It belongs on the engine's compute, with this
  workspace reading the output.

## If the outcome is GO

Compare two nets, one change between them:

- **A:** net8x160's recipe as it is.
- **B:** the same recipe plus a narrow set of sparse pawn-pair inputs, chosen to cover
  the cells that met the size condition.

Hold the data, the label sets, the width, the schedule and the seed policy. Measure
the cost of updating the new inputs and the nodes per second first. Then test at
fixed nodes and at a clock by SPRT, with the correction history on, as shipped. Only
that result says whether the inputs make the engine stronger.

## If the outcome is NO-GO

Record it in the engine's `docs/research-log.md` with the `MIN_EFFECT_CP` that was
ruled out and the sample it was ruled out on, so the idea is not proposed again
without new evidence.
