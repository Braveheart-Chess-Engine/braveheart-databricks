"""The structural analysis's statistics, on the driver: a logistic model of the game's
result from the net's evaluation, with and without the pawn-structure differences.

The rows are structural_frames.model_rows': one per kept position, with its game, its
worker group's fold, its phase, the evaluation and each feature as white's count minus
black's. The result is 0, a half or 1 from white's point of view, so the log loss is the
cross-entropy against a soft label and a prediction is an expected score.

Three things are measured, per generation (`docs/nnue-structural-research.md`):

- the **gain** in log loss from adding the structure terms, on predictions each position
  gets from a model fitted without its fold, with an interval from resampling games;
- the same gain with the structure **shuffled** across positions, the control;
- each structure term's **effect** in centipawns of evaluation, from one fit on every row,
  with standard errors clustered by game.

Everything is NumPy and pandas, so it runs in tests without Spark.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .phase import PHASES
from .registered import BOOTSTRAP, EVAL_CLIP, RIDGE
from .structure import STRUCTURE

_BASELINE = ("eval", "npm_diff", "pawns_d")
_CHUNK = 200_000
_EPSILON = 1e-12


def design(rows: pd.DataFrame, structure: bool) -> tuple[np.ndarray, list[str]]:
    """The model's matrix and its column names: `intercept`, `stm` (+1 with white to
    move, -1 with black), then per phase `eval:<phase>` (the evaluation, clipped to
    ±EVAL_CLIP, in units of 100 cp), `npm_diff:<phase>` and `pawns_d:<phase>`, and with
    `structure` also `<feature>:<phase>` for each structure difference. A column is zero
    outside its phase, so every term has its own coefficient per phase; the baseline's
    columns come first."""
    n = len(rows)
    phase = rows["phase"].to_numpy()
    terms = {
        "eval": np.clip(rows["eval_white"].to_numpy(dtype=np.float64), -EVAL_CLIP, EVAL_CLIP) / 100,
        "npm_diff": rows["npm_diff"].to_numpy(dtype=np.float64),
        "pawns_d": rows["pawns_d"].to_numpy(dtype=np.float64),
    }
    names = ["intercept", "stm"]
    columns = [np.ones(n), rows["stm"].to_numpy(dtype=np.float64)]
    inside = {name: phase == name for name in PHASES}
    for name in PHASES:
        for term in _BASELINE:
            names.append(f"{term}:{name}")
            columns.append(np.where(inside[name], terms[term], 0.0))
    if structure:
        for feature in STRUCTURE:
            values = rows[f"{feature}_d"].to_numpy(dtype=np.float64)
            for name in PHASES:
                names.append(f"{feature}:{name}")
                columns.append(np.where(inside[name], values, 0.0))
    return np.column_stack(columns).astype(np.float32), names


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _moments(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The predictions, the gradient X'(p - y) and the Hessian X' diag(p(1 - p)) X, a
    chunk of rows at a time."""
    k = X.shape[1]
    p = np.empty(len(X))
    gradient, hessian = np.zeros(k), np.zeros((k, k))
    for start in range(0, len(X), _CHUNK):
        part = X[start : start + _CHUNK].astype(np.float64)
        prob = _sigmoid(part @ beta)
        p[start : start + _CHUNK] = prob
        gradient += part.T @ (prob - y[start : start + _CHUNK])
        hessian += (part * (prob * (1 - prob))[:, None]).T @ part
    return p, gradient, hessian


def fit(X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """The coefficients that minimise the log loss of sigmoid(X beta) against `y` (0, a
    half or 1) plus `l2` / 2 times the squared coefficients, by Newton's method. The first
    column is the intercept, which the penalty skips. A column that is zero in every row
    has no coefficient (nan) and is left out."""
    y = np.asarray(y, dtype=np.float64)
    active = np.flatnonzero(np.any(X != 0, axis=0))
    part = X[:, active]
    penalty = np.full(len(active), l2)
    penalty[active == 0] = 0.0
    beta = np.zeros(len(active))
    for _ in range(100):
        _, gradient, hessian = _moments(part, y, beta)
        gradient += penalty * beta
        hessian[np.diag_indices_from(hessian)] += penalty + _EPSILON
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step), initial=0.0) < 1e-10:
            break
    out = np.full(X.shape[1], np.nan)
    out[active] = beta
    return out


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """The expected result of each row; a column without a coefficient adds nothing."""
    weights = np.nan_to_num(beta)
    out = np.empty(len(X))
    for start in range(0, len(X), _CHUNK):
        out[start : start + _CHUNK] = _sigmoid(
            X[start : start + _CHUNK].astype(np.float64) @ weights
        )
    return out


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each row's cross-entropy between the result `y` (0, a half or 1) and the expected
    result `p`. At a drawn game it is least, log 2, when p is a half."""
    p = np.clip(p, _EPSILON, 1 - _EPSILON)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cross_fitted(rows: pd.DataFrame, structure: bool, l2: float = RIDGE) -> np.ndarray:
    """Each row's expected result from a model fitted on the other folds: no worker group
    is on both sides of its own prediction."""
    X, _ = design(rows, structure)
    y = rows["result"].to_numpy(dtype=np.float64)
    folds = rows["fold"].to_numpy()
    values = np.unique(folds)
    if len(values) < 2:
        raise ValueError("cross-fitting needs at least two folds of worker groups")
    out = np.empty(len(rows))
    for fold in values:
        inside = folds == fold
        out[inside] = predict(X[inside], fit(X[~inside], y[~inside], l2))
    return out


def shuffle_structure(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The rows with their structure differences permuted across positions within cells of
    phase and evaluation (100 cp wide). A position's seven differences move together, so
    their joint distribution in every cell is kept and their link to the result is
    broken."""
    rng = np.random.default_rng(seed)
    cell = pd.factorize(
        pd.MultiIndex.from_arrays([rows["phase"], rows["eval_white"] // 100]), sort=True
    )[0]
    original = np.lexsort((np.arange(len(rows)), cell))
    shuffled = np.lexsort((rng.random(len(rows)), cell))
    out = rows.copy()
    for feature in STRUCTURE:
        column = f"{feature}_d"
        values = rows[column].to_numpy()
        moved = np.empty_like(values)
        moved[original] = values[shuffled]
        out[column] = moved
    return out


def _by_game(rows: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Each row's game as a code 0..G-1, and G."""
    codes, uniques = pd.factorize(rows["game_id"])
    return codes, len(uniques)


def gain(
    rows: pd.DataFrame,
    p_base: np.ndarray,
    p_other: np.ndarray,
    n_boot: int = BOOTSTRAP,
    seed: int = 0,
) -> dict[str, float]:
    """How much lower the log loss of `p_other` is than `p_base`'s, per position (positive:
    the other model predicts better): the `mean`, and the 95% interval `lo` to `hi` from
    resampling whole games `n_boot` times."""
    y = rows["result"].to_numpy(dtype=np.float64)
    difference = log_loss(p_base, y) - log_loss(p_other, y)
    codes, games = _by_game(rows)
    total = np.bincount(codes, weights=difference, minlength=games)
    count = np.bincount(codes, minlength=games).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, games, games)
        means[i] = total[picked].sum() / count[picked].sum()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(total.sum() / count.sum()), "lo": float(lo), "hi": float(hi)}


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def effects(rows: pd.DataFrame, min_effect_cp: float) -> pd.DataFrame:
    """Each structure feature's effect in each phase, from one fit on every row with no
    penalty: `cp`, how many centipawns the evaluation would have to move per unit of the
    feature's white-minus-black difference to predict the results it missed (positive: the
    side with more of the feature scores better than the evaluation says); its standard
    error `se`, clustered by game; the 95% interval `lo` to `hi`; `share`, the phase's
    rows where the sides differ in the feature; and `power`, the chance the interval would
    exclude zero were the effect `min_effect_cp`. A feature with no variation in a phase,
    or a phase with no rows, has share 0 and no effect (nan)."""
    X, names = design(rows, structure=True)
    y = rows["result"].to_numpy(dtype=np.float64)
    beta = fit(X, y)
    active = np.flatnonzero(~np.isnan(beta))
    part = X[:, active]
    p, _, hessian = _moments(part, y, beta[active])
    # The sandwich: the scores x (p - y), summed within each game, are the independent
    # units. Rows of one game share its result, so their scores are far from independent.
    codes, games = _by_game(rows)
    residual = p - y
    by_game = np.zeros((games, len(active)))
    for start in range(0, len(rows), _CHUNK):
        chunk = slice(start, start + _CHUNK)
        np.add.at(by_game, codes[chunk], part[chunk].astype(np.float64) * residual[chunk, None])
    meat = by_game.T @ by_game * (games / max(games - 1, 1))
    bread = np.linalg.inv(hessian + _EPSILON * np.eye(len(active)))
    covariance = bread @ meat @ bread
    position = {names[column]: i for i, column in enumerate(active)}
    phase = rows["phase"].to_numpy()
    out = []
    for feature in STRUCTURE:
        differs = rows[f"{feature}_d"].to_numpy() != 0
        for name in PHASES:
            inside = phase == name
            share = float(differs[inside].mean()) if inside.any() else 0.0
            b, a = position.get(f"{feature}:{name}"), position.get(f"eval:{name}")
            if b is None or a is None or beta[active][a] == 0:
                out.append((feature, name, np.nan, np.nan, np.nan, np.nan, share, np.nan))
                continue
            slope = beta[active][a]
            ratio = beta[active][b] / slope
            variance = (100 / slope) ** 2 * (
                covariance[b, b] - 2 * ratio * covariance[a, b] + ratio**2 * covariance[a, a]
            )
            cp, se = 100 * ratio, math.sqrt(max(variance, 0.0))
            power = _normal_cdf(min_effect_cp / se - 1.96) if se > 0 else 1.0
            out.append((feature, name, cp, se, cp - 1.96 * se, cp + 1.96 * se, share, power))
    return pd.DataFrame(out, columns=["feature", "phase", "cp", "se", "lo", "hi", "share", "power"])


def duplicate_share(rows: pd.DataFrame) -> float:
    """The share of rows whose position (board and side to move) is also in another fold:
    how much the folds overlap in exact positions, though never in games."""
    folds = rows.groupby("position_hash")["fold"].transform("nunique")
    return float((folds > 1).mean())


def difference_shares(summary: pd.DataFrame) -> pd.DataFrame:
    """Per generation, phase and structure feature, the positions counted and the share of
    them where the sides differ in the feature, from the summary's `feature` rows
    (structural_frames.summary): how often each feature has anything to say."""
    rows = summary[summary["kind"] == "feature"]
    keys = ["gen", "phase", "feature"]
    positions = rows.groupby(keys, observed=True)["n"].sum()
    differing = rows[rows["value"] != 0].groupby(keys, observed=True)["n"].sum()
    out = positions.rename("positions").to_frame()
    out["share"] = (differing.reindex(out.index).fillna(0) / out["positions"]).astype(float)
    return out.reset_index().astype({"gen": int, "positions": int})
