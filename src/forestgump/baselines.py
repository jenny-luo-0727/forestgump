"""The three baselines, their small grids, and grid selection on training rows only."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .featurize import Fingerprints
from .stats import LOWER_IS_BETTER, PRIMARY, metrics

N_TREES = 500
METHODS = ("mean", "knn", "forest")

GRIDS: dict[str, list[dict]] = {
    "mean": [{}],
    "knn": [
        {"radius": r, "n_neighbors": k, "weights": w}
        for r, k, w in itertools.product((2, 3), (1, 3, 5, 10), ("uniform", "distance"))
    ],
    "forest": [
        {"radius": r, "n_estimators": N_TREES, "max_features": mf, "min_samples_leaf": leaf}
        for r, mf, leaf in itertools.product((2, 3), ("sqrt", 0.3, 1.0), (1, 3))
    ],
}


def _model(method: str, task: str, params: dict, n_train: int, seed: int, n_jobs: int):
    binary = task == "binary"
    if method == "mean":
        return DummyClassifier(strategy="prior") if binary else DummyRegressor(strategy="mean")
    if method == "knn":
        cls = KNeighborsClassifier if binary else KNeighborsRegressor
        return cls(n_neighbors=min(params["n_neighbors"], n_train), weights=params["weights"], metric="precomputed")
    cls = RandomForestClassifier if binary else RandomForestRegressor
    return cls(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=seed,
        n_jobs=n_jobs,
    )


def tanimoto_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - Tanimoto similarity between the rows of two bit matrices (scipy's Jaccard distance; 0 if both empty)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    shared = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - shared
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = (union - shared) / union  # integer counts, so this rounds exactly as scipy's jaccard does
    distance[union == 0] = 0.0
    return distance


def fingerprints_for(fps: Fingerprints, params: dict) -> np.ndarray | None:
    """The bit matrix a config uses (all rows), or None for the mean baseline."""
    return fps(params["radius"]) if "radius" in params else None


def _features(method: str, X: np.ndarray | None, train_idx: np.ndarray, test_idx: np.ndarray):
    """(X to fit on, X to predict). kNN gets precomputed Tanimoto distances to the training rows."""
    if X is None:
        # The mean baseline ignores X; one zero column keeps sklearn's input checks happy.
        return np.zeros((len(train_idx), 1)), np.zeros((len(test_idx), 1))
    if method == "knn":
        return tanimoto_distance(X[train_idx], X[train_idx]), tanimoto_distance(X[test_idx], X[train_idx])
    return X[train_idx], X[test_idx]


def predict(model, task: str, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if task == "binary" else model.predict(X)


def fit_predict(method, task, params, X, y, train_idx, test_idx, seed, n_jobs):
    """Fit one configuration on rows train_idx of X and predict rows test_idx. Returns (predictions, fitted model).

    X is the config's fingerprint matrix over all rows (`fingerprints_for`), or None for the mean baseline.
    """
    model = _model(method, task, params, len(train_idx), seed, n_jobs)
    X_fit, X_pred = _features(method, X, train_idx, test_idx)
    model.fit(X_fit, y[train_idx])
    return predict(model, task, X_pred), model


def _inner_cv_score(method, task, params, X, y, train_idx, seed) -> float:
    """Mean primary metric over inner CV folds of the training rows."""
    scores = []
    for a, b in _inner_splits(task, y[train_idx], seed).split(train_idx, y[train_idx]):
        p, _ = fit_predict(method, task, params, X, y, train_idx[a], train_idx[b], seed, n_jobs=1)
        scores.append(metrics(task, y[train_idx[b]], p)[PRIMARY[task]])
    return float(np.mean(scores))


def _inner_splits(task: str, y_train: np.ndarray, seed: int):
    n_splits = 5 if len(y_train) >= 50 else 3
    if task == "binary":
        n_splits = min(n_splits, int(np.bincount(y_train.astype(int)).min()))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _fit_forest_with_oob(task, params, X, y, train_idx, seed, n_jobs):
    """Fit one forest on the training rows; score it on its out-of-bag predictions. Returns (score, model)."""
    model = _model("forest", task, params, len(train_idx), seed, n_jobs).set_params(oob_score=True)
    model.fit(X[train_idx], y[train_idx])
    oob = model.oob_decision_function_[:, 1] if task == "binary" else model.oob_prediction_
    seen = np.isfinite(oob)  # a row in every bootstrap sample has no out-of-bag prediction
    return metrics(task, y[train_idx][seen], oob[seen])[PRIMARY[task]], model


def _better(primary: str, a: float, b: float) -> bool:
    return a < b if primary in LOWER_IS_BETTER else a > b


@dataclass
class Selection:
    """The grid winner for one method on one cell, with every config's score on the training rows."""

    params: dict
    score: float
    grid_scores: list[float] = field(default_factory=list)
    model: object = None  # the forest winner, already fitted on every training row


def select(method, task, fps, y, train_idx, seed, n_jobs) -> Selection:
    """Pick a method's config using only the rows in train_idx.

    The forest fits each of its configs once on all training rows and is judged by its out-of-bag score, which
    costs nothing extra and needs no inner CV; the winning forest is kept, so nothing is refit. kNN fits take
    milliseconds, so it uses inner CV. Ties go to the earlier config in grid order, so selection is deterministic.
    """
    grid = GRIDS[method]
    primary = PRIMARY[task]
    if len(grid) == 1:
        return Selection(grid[0], np.nan, [np.nan])
    best, scores = None, []
    for params in grid:
        X = fingerprints_for(fps, params)
        if method == "forest":
            score, model = _fit_forest_with_oob(task, params, X, y, train_idx, seed, n_jobs)
        else:
            score, model = _inner_cv_score(method, task, params, X, y, train_idx, seed), None
        scores.append(score)
        if best is None or _better(primary, score, best.score):
            best = Selection(params, score, model=model)
    best.grid_scores = scores
    return best
