import numpy as np
import pandas as pd
import pytest

from selfplay_analysis.phase import PHASES
from selfplay_analysis.registered import FOLDS
from selfplay_analysis.structural_model import (
    cross_fitted,
    design,
    duplicate_share,
    effects,
    fit,
    gain,
    log_loss,
    predict,
    shuffle_structure,
)
from selfplay_analysis.structure import STRUCTURE


def make_rows(
    n_games: int,
    per_game: int,
    effect_cp: float,
    seed: int,
    noise: float = 0.0,
    with_feature: float = 0.0,
) -> pd.DataFrame:
    """Model rows as structural_frames.model_rows returns them. Per game: a fold, a phase,
    a difference for each structure feature and an evaluation, which its rows share unless
    `noise` (centipawns) spreads them around it; `with_feature` adds that many centipawns
    of evaluation per isolated pawn of difference. One result per game, drawn (with draws)
    so that its expectation is sigmoid((eval + effect_cp * isolated_d) / 400) at the
    game's first row: without noise the model fitted is the one the data came from."""
    rng = np.random.default_rng(seed)
    game = np.repeat(np.arange(n_games), per_game)
    differences = {feature: rng.integers(-2, 3, n_games) for feature in STRUCTURE}
    level = rng.normal(0, 150, n_games) + with_feature * differences["isolated"]
    evals = np.round(level[game] + rng.normal(0, noise, len(game)))
    first = evals.reshape(n_games, per_game)[:, 0]
    expected = 1 / (1 + np.exp(-(first + effect_cp * differences["isolated"]) / 400))
    draw = np.minimum(expected, 1 - expected)
    u = rng.random(n_games)
    result = np.where(u < expected - draw / 2, 1.0, np.where(u < expected + draw / 2, 0.5, 0.0))
    rows = pd.DataFrame(
        {
            "fold": rng.integers(0, FOLDS, n_games)[game],
            "game_id": game,
            "phase": rng.choice(PHASES, n_games)[game],
            "stm": rng.choice([-1, 1], len(game)),
            "eval_white": evals.astype(int),
            "result": result[game],
            "npm_diff": rng.integers(-3, 4, len(game)),
            "pawns_d": rng.integers(-1, 2, len(game)),
            "position_hash": rng.integers(0, 2**62, len(game)),
        }
    )
    for feature in STRUCTURE:
        rows[f"{feature}_d"] = differences[feature][game]
    return rows


def test_the_design_has_a_column_per_term_and_phase() -> None:
    rows = make_rows(50, 2, effect_cp=0.0, seed=0)
    base, base_names = design(rows, structure=False)
    full, names = design(rows, structure=True)
    assert base_names == ["intercept", "stm"] + [
        f"{term}:{phase}" for phase in PHASES for term in ("eval", "npm_diff", "pawns_d")
    ]
    assert names == base_names + [f"{f}:{phase}" for f in STRUCTURE for phase in PHASES]
    assert base.shape == (100, 11) and full.shape == (100, 32)
    first = rows.iloc[0]
    row = dict(zip(names, full[0].tolist(), strict=True))
    assert row["intercept"] == 1.0 and row["stm"] == first["stm"]
    for phase in PHASES:
        inside = phase == first["phase"]
        assert row[f"eval:{phase}"] == pytest.approx(first["eval_white"] / 100 if inside else 0.0)
        assert row[f"passed:{phase}"] == (first["passed_d"] if inside else 0.0)


def test_the_design_is_single_precision() -> None:
    matrix, _ = design(make_rows(50, 2, effect_cp=0.0, seed=0), structure=True)
    assert matrix.dtype == np.float32 and matrix.flags["C_CONTIGUOUS"]


def test_the_evaluation_is_clipped() -> None:
    rows = make_rows(4, 1, effect_cp=0.0, seed=0)
    rows["phase"] = "endgame"
    rows["eval_white"] = [5000, -5000, 300, 0]
    matrix, names = design(rows, structure=False)
    assert matrix[:, names.index("eval:endgame")].tolist() == [20.0, -20.0, 3.0, 0.0]


def test_fit_recovers_known_coefficients_from_soft_labels() -> None:
    X = np.column_stack([np.ones(500), np.linspace(-3, 3, 500)])
    beta = np.array([0.2, -0.7])
    assert np.allclose(fit(X, 1 / (1 + np.exp(-X @ beta))), beta, atol=1e-6)


def test_a_penalty_shrinks_everything_but_the_intercept() -> None:
    X = np.column_stack([np.ones(500), np.linspace(-3, 3, 500)])
    y = 1 / (1 + np.exp(-X @ np.array([0.2, -0.7])))
    plain, shrunk = fit(X, y), fit(X, y, l2=500.0)
    assert abs(shrunk[1]) < abs(plain[1]) / 2
    assert shrunk[0] == pytest.approx(plain[0], abs=0.05)


def test_an_empty_column_has_no_coefficient_and_predicts_nothing() -> None:
    X = np.column_stack([np.ones(500), np.linspace(-3, 3, 500), np.zeros(500)])
    beta = fit(X, 1 / (1 + np.exp(-X[:, :2] @ np.array([0.2, -0.7]))))
    assert np.isnan(beta[2]) and np.allclose(beta[:2], [0.2, -0.7], atol=1e-6)
    assert np.allclose(predict(X, beta), 1 / (1 + np.exp(-X[:, :2] @ beta[:2])))


def test_log_loss_is_the_cross_entropy_against_a_soft_result() -> None:
    losses = log_loss(np.array([0.5, 0.8, 0.8]), np.array([0.5, 1.0, 0.0]))
    assert np.allclose(losses, [np.log(2), -np.log(0.8), -np.log(0.2)])


def test_a_planted_effect_is_recovered_in_centipawns() -> None:
    rows = make_rows(4000, 6, effect_cp=0.0, seed=1)
    # +15 cp an isolated pawn: 0.25 a pawn of evaluation, 0.25 * 0.15 an isolated pawn.
    logit = 0.25 * rows["eval_white"] / 100 + 0.25 * 0.15 * rows["isolated_d"]
    rows["result"] = 1 / (1 + np.exp(-logit))
    table = effects(rows).set_index(["feature", "phase"])
    assert np.allclose(table.xs("isolated", level="feature")["cp"], 15.0, atol=0.01)
    assert np.allclose(table.xs("doubled", level="feature")["cp"], 0.0, atol=0.01)


def test_effects_have_a_row_per_feature_and_phase() -> None:
    table = effects(make_rows(600, 4, effect_cp=0.0, seed=2))
    assert list(table.columns) == ["feature", "phase", "cp", "se", "lo", "hi", "share"]
    assert list(zip(table["feature"], table["phase"], strict=True)) == [
        (feature, phase) for feature in STRUCTURE for phase in PHASES
    ]
    assert np.allclose(table["lo"], table["cp"] - 1.96 * table["se"])
    assert np.allclose(table["hi"], table["cp"] + 1.96 * table["se"])
    # Differences are drawn from -2..2, so the sides differ in four games of five.
    assert table["share"].between(0.7, 0.9).all()


def test_the_interval_of_a_large_effect_matches_refitting_resampled_games() -> None:
    # The effect is a ratio, so its error has three terms: the feature coefficient's, the
    # evaluation coefficient's (which matters when the ratio is large) and their
    # covariance (which matters when the feature goes with the evaluation). Here both do:
    # 150 cp an isolated pawn, and 40 cp of evaluation with each. Refitting on games drawn
    # with replacement measures the same error with no formula.
    rows = make_rows(3000, 2, effect_cp=150.0, seed=11, with_feature=40.0)
    table = effects(rows).set_index(["feature", "phase"])
    games = rows["game_id"].to_numpy().reshape(-1, 2)[:, 0]
    by_game = rows.set_index("game_id")
    rng = np.random.default_rng(0)
    refits = []
    for _ in range(120):
        picked = rng.choice(games, len(games))
        sample = by_game.loc[picked].reset_index(drop=True)
        sample["game_id"] = np.repeat(np.arange(len(picked)), 2)  # a game drawn twice is two games
        refits.append(effects(sample).set_index(["feature", "phase"])["cp"])
    spread = pd.concat(refits, axis=1).std(axis=1)
    planted = [("isolated", phase) for phase in PHASES]
    assert (table.loc[planted, "se"] / spread.loc[planted]).between(0.8, 1.25).all()
    assert ((table.loc[planted, "cp"] - 150).abs() < 4 * table.loc[planted, "se"]).all()


def test_clustered_errors_hold_their_level_when_nothing_is_there() -> None:
    # One result per game, features constant within a game, evaluations spread around the
    # game's level (so the evaluation term is not even the true one): the intervals of the
    # 21 effects, none of which is real, should exclude zero about one time in twenty.
    excluded = []
    for seed in range(100):
        table = effects(make_rows(3000, 4, effect_cp=0.0, seed=seed, noise=60.0))
        excluded.append(((table["lo"] > 0) | (table["hi"] < 0)).to_numpy())
    assert 0.03 <= np.mean(excluded) <= 0.07


def test_a_real_effect_is_found_where_it_was_put() -> None:
    table = effects(make_rows(30_000, 2, effect_cp=60.0, seed=8)).set_index(["feature", "phase"])
    planted = table.xs("isolated", level="feature")
    assert (planted["lo"] > 0).all() and planted["cp"].between(40, 80).all()
    others = table.drop("isolated", level="feature")
    assert (others["cp"].abs() < 4 * others["se"]).all()


def test_a_feature_absent_from_a_phase_is_missing_not_fatal() -> None:
    rows = make_rows(800, 6, effect_cp=0.0, seed=3)
    rows.loc[rows["phase"] == "opening", "protected_passed_d"] = 0
    table = effects(rows).set_index(["feature", "phase"])
    assert np.isnan(table.loc[("protected_passed", "opening"), "cp"])
    assert table.loc[("protected_passed", "opening"), "share"] == 0.0
    assert np.isfinite(table.loc[("isolated", "opening"), "cp"])


def test_a_phase_with_no_positions_is_missing_not_fatal() -> None:
    rows = make_rows(800, 6, effect_cp=0.0, seed=3)
    rows = rows[rows["phase"] != "opening"].reset_index(drop=True)
    table = effects(rows).set_index(["feature", "phase"])
    assert table.xs("opening", level="phase")["cp"].isna().all()
    assert (table.xs("opening", level="phase")["share"] == 0.0).all()
    assert np.isfinite(table.loc[("isolated", "endgame"), "cp"])


def test_cross_fitting_never_predicts_a_row_with_its_own_fold() -> None:
    rows = make_rows(500, 4, effect_cp=20.0, seed=4)
    X, _ = design(rows, structure=True)
    y = rows["result"].to_numpy()
    inside = (rows["fold"] == 2).to_numpy()
    expected = predict(X[inside], fit(X[~inside], y[~inside], l2=1.0))
    assert np.allclose(cross_fitted(rows, True)[inside], expected)


def test_cross_fitting_needs_two_folds() -> None:
    rows = make_rows(50, 2, effect_cp=0.0, seed=4)
    rows["fold"] = 0
    with pytest.raises(ValueError, match="two folds"):
        cross_fitted(rows, structure=False)


def test_shuffling_keeps_each_cells_values_and_breaks_the_order() -> None:
    rows = make_rows(500, 4, effect_cp=0.0, seed=5)
    mixed = shuffle_structure(rows, seed=0)
    cell = [rows["phase"], rows["eval_white"] // 100]
    before = rows.groupby(cell)["isolated_d"].apply(sorted)
    assert before.equals(mixed.groupby(cell)["isolated_d"].apply(sorted))
    assert not rows["isolated_d"].equals(mixed["isolated_d"])
    assert rows["result"].equals(mixed["result"]) and rows["eval_white"].equals(mixed["eval_white"])
    # A position's features move together, and the same seed moves them the same way.
    pairs = set(zip(rows["isolated_d"], rows["passed_d"], rows["phase"], strict=True))
    assert set(zip(mixed["isolated_d"], mixed["passed_d"], mixed["phase"], strict=True)) <= pairs
    assert mixed.equals(shuffle_structure(rows, seed=0))


def test_gain_is_zero_for_identical_predictions_and_positive_for_a_real_effect() -> None:
    rows = make_rows(20_000, 2, effect_cp=60.0, seed=6)
    base, full = cross_fitted(rows, False), cross_fitted(rows, True)
    assert gain(rows, base, base) == {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    found = gain(rows, base, full)
    assert found["lo"] > 0.0 and found["lo"] <= found["mean"] <= found["hi"]
    assert found == gain(rows, base, full)  # seeded


def test_the_gains_interval_resamples_games_not_positions() -> None:
    # Eight identical positions a game: the games, not their rows, are the evidence. The
    # interval must be as wide as the games' own means say, which is about the square root
    # of eight times wider than resampling rows would make it.
    rows = make_rows(1500, 8, effect_cp=60.0, seed=9)
    base, full = cross_fitted(rows, False), cross_fitted(rows, True)
    found = gain(rows, base, full)
    y = rows["result"].to_numpy()
    per_game = pd.Series(log_loss(base, y) - log_loss(full, y)).groupby(rows["game_id"]).mean()
    expected = per_game.std() / np.sqrt(len(per_game))
    assert (found["hi"] - found["lo"]) / 3.92 == pytest.approx(expected, rel=0.1)


def test_shuffled_features_gain_nothing() -> None:
    rows = make_rows(20_000, 2, effect_cp=60.0, seed=6)
    mixed = shuffle_structure(rows, seed=0)
    found = gain(mixed, cross_fitted(mixed, False), cross_fitted(mixed, True))
    assert found["hi"] < gain(rows, cross_fitted(rows, False), cross_fitted(rows, True))["mean"]


def test_duplicate_share_counts_positions_seen_in_more_than_one_fold() -> None:
    rows = pd.DataFrame({"position_hash": [7, 7, 8, 9], "fold": [0, 1, 0, 0]})
    assert duplicate_share(rows) == 0.5
    rows["fold"] = [0, 0, 0, 0]
    assert duplicate_share(rows) == 0.0
