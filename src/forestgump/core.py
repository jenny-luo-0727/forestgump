"""Run every baseline on every cell and compare it with the user's model."""

from __future__ import annotations

import time
import warnings

import pandas as pd

from .baselines import FOREST_SELECTION, METHODS, fingerprints_for, fit_predict, predict, select
from .featurize import Fingerprints
from .io import load
from .report import Comparison, Report
from .stats import MIN_CELLS, PRIMARY, corrected_t, holm, improvement, metrics

INNER_CV_WARNING = (
    "forest_selection='inner-cv' fits every forest config once per inner fold, then refits the winner: "
    "about 5x slower than the default out-of-bag selection. The forest will take its time."
)


def run(
    df: pd.DataFrame,
    *,
    smiles: str = "smiles",
    target: str = "target",
    fold: str = "fold",
    pred: str = "pred",
    repeat: str | None = None,
    task: str | None = None,
    seed: int = 0,
    n_jobs: int = -1,
    alpha: float = 0.05,
    forest_selection: str = "oob",
) -> Report:
    """Fit the mean, kNN and forest baselines on the user's folds and test the user's model against each.

    See `forestgump.io.load` for the input contract. Hyperparameters are chosen per cell from that cell's training
    rows only; test rows are only ever predicted. kNN always uses inner CV. The forest uses its out-of-bag score
    by default (one fit per config); forest_selection="inner-cv" uses inner CV instead, at about 5x the time.
    """
    if forest_selection not in FOREST_SELECTION:
        raise ValueError(f"forest_selection must be one of {FOREST_SELECTION}, got {forest_selection!r}")
    if forest_selection == "inner-cv":
        warnings.warn(INNER_CV_WARNING, stacklevel=2)
    ds = load(df, smiles=smiles, target=target, fold=fold, pred=pred, repeat=repeat, task=task)
    fps = Fingerprints(ds.mols, ds.mol_index)
    rows, selections, predictions = [], [], []
    forest_seconds, n_leaves = 0.0, 0
    for i, cell in enumerate(ds.cells):
        base = {"cell": i, "repeat": cell.repeat, "fold": cell.fold, "n_train": len(cell.train_idx)}
        base["n_test"] = len(cell.test_idx)
        rows.append(base | {"method": "user"} | metrics(ds.task, ds.y[cell.test_idx], ds.pred[cell.test_idx]))
        for method in METHODS:
            start = time.perf_counter()
            sel = select(method, ds.task, fps, ds.y, cell.train_idx, seed, n_jobs, forest_selection)
            if sel.model is None:
                X = fingerprints_for(fps, sel.params)
                p, model = fit_predict(
                    method, ds.task, sel.params, X, ds.y, cell.train_idx, cell.test_idx, seed, n_jobs
                )
            else:
                model = sel.model
                p = predict(model, ds.task, fingerprints_for(fps, sel.params)[cell.test_idx])
            if method == "forest":
                forest_seconds += time.perf_counter() - start
                n_leaves = sum(int(t.get_n_leaves()) for t in model.estimators_)
            rows.append(base | {"method": method} | metrics(ds.task, ds.y[cell.test_idx], p))
            predictions.append(pd.DataFrame({"cell": i, "row": cell.test_idx, "method": method, "pred": p}))
            selections.append({"cell": i, "method": method, "params": sel.params, "selection_score": sel.score})

    table = pd.DataFrame(rows)
    primary = PRIMARY[ds.task]
    wide = table.pivot(index="cell", columns="method", values=primary)
    comparisons = []
    if len(ds.cells) >= MIN_CELLS:
        tests = [
            corrected_t(improvement(primary, wide["user"].to_numpy(), wide[m].to_numpy()), ds.test_train_ratio, alpha)
            for m in METHODS
        ]
        for m, t, p_adj in zip(METHODS, tests, holm([t.p for t in tests]), strict=True):
            comparisons.append(Comparison(method=m, test=t, p_holm=p_adj))
    return Report(
        task=ds.task,
        primary=primary,
        table=table,
        selections=selections,
        predictions=pd.concat(predictions, ignore_index=True),
        comparisons=comparisons,
        n_cells=len(ds.cells),
        test_train_ratio=ds.test_train_ratio,
        forest_seconds=forest_seconds,
        forest_leaves=n_leaves,
        alpha=alpha,
        seed=seed,
        classes=ds.classes,
        forest_selection=forest_selection,
    )
