"""Load the user's molecules, folds and predictions, and refuse anything that would make the comparison lie."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

TASKS = ("regression", "binary")
MIN_TRAIN_ROWS = 6


class InputError(ValueError):
    """The input cannot be compared fairly. The message says which rows and why."""


@dataclass(frozen=True)
class Cell:
    """One (repeat, fold) test set: the paired unit of every comparison."""

    repeat: object
    fold: object
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class Dataset:
    """Validated input. Row order follows the input frame; `mol_index` maps each row to `mols`."""

    task: str
    y: np.ndarray
    pred: np.ndarray
    mols: tuple
    mol_index: np.ndarray
    cells: tuple[Cell, ...]
    classes: tuple | None = None

    @property
    def test_train_ratio(self) -> float:
        """Mean n_test / n_train over cells, for the Nadeau-Bengio correction."""
        return float(np.mean([len(c.test_idx) / len(c.train_idx) for c in self.cells]))


def _rows(mask: np.ndarray, limit: int = 10) -> str:
    """Name offending rows as 0-based positions in the input frame."""
    rows = np.flatnonzero(mask)
    shown = ", ".join(str(r) for r in rows[:limit])
    return shown + (f", ... ({len(rows)} rows)" if len(rows) > limit else "")


def _parse_smiles(smiles: np.ndarray) -> tuple[tuple, np.ndarray]:
    unique, inverse = np.unique(smiles.astype(str), return_inverse=True)
    RDLogger.DisableLog("rdApp.*")
    try:
        mols = [Chem.MolFromSmiles(s) for s in unique]
    finally:
        RDLogger.EnableLog("rdApp.*")
    bad = np.array([m is None for m in mols])
    if bad.any():
        raise InputError(f"RDKit could not parse SMILES in rows {_rows(bad[inverse])}: {list(unique[bad][:5])}")
    return tuple(mols), inverse


def _resolve_task(target: pd.Series, task: str | None) -> tuple[str, np.ndarray, tuple | None]:
    n_unique = target.nunique()
    if task is None:
        task = "binary" if n_unique == 2 else "regression"
    if task not in TASKS:
        raise InputError(f"task must be one of {TASKS}, got {task!r}")
    if task == "binary":
        if n_unique != 2:
            raise InputError(f"task='binary' needs exactly 2 distinct target values, found {n_unique}")
        classes = tuple(sorted(target.unique()))
        return task, (target.to_numpy() == classes[1]).astype(float), classes
    try:
        y = pd.to_numeric(target, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError(f"task='regression' needs a numeric target: {exc}") from None
    return task, y, None


def _build_cells(fold: pd.Series, repeat: pd.Series) -> tuple[Cell, ...]:
    cells = []
    for r in sorted(repeat.unique(), key=str):
        in_repeat = (repeat == r).to_numpy()
        folds = sorted(fold[in_repeat].unique(), key=str)
        if len(folds) < 2:
            raise InputError(f"repeat {r!r} has a single fold ({folds[0]!r}), so it has no training set")
        for f in folds:
            test = in_repeat & (fold == f).to_numpy()
            cells.append(Cell(r, f, np.flatnonzero(in_repeat & ~test), np.flatnonzero(test)))
    return tuple(cells)


def _check_cells(cells: tuple[Cell, ...], task: str, y: np.ndarray) -> None:
    for c in cells:
        where = f"repeat {c.repeat!r}, fold {c.fold!r}"
        if len(c.train_idx) < MIN_TRAIN_ROWS:
            raise InputError(
                f"{where} has {len(c.train_idx)} training rows; the forest needs at least {MIN_TRAIN_ROWS} "
                "(it is a forest, not a shrub)"
            )
        if task == "binary":
            train_counts = np.bincount(y[c.train_idx].astype(int), minlength=2)
            if train_counts.min() < 2:
                raise InputError(f"{where}: the training set needs at least 2 rows of each class, has {train_counts}")
            if len(np.unique(y[c.test_idx])) < 2:
                raise InputError(f"{where}: the test set has one class, so ROC-AUC is undefined. Use stratified folds.")


def load(
    df: pd.DataFrame,
    *,
    smiles: str = "smiles",
    target: str = "target",
    fold: str = "fold",
    pred: str = "pred",
    repeat: str | None = None,
    task: str | None = None,
) -> Dataset:
    """Validate a frame with one row per (molecule, repeat) and return the paired test cells.

    The test set of cell (r, f) is the rows with ``repeat == r`` and ``fold == f``; its training set is every
    other row of repeat r. `pred` holds the user's model's prediction for the row when it was in the test set:
    a value for regression, the probability of the larger class label for binary.
    """
    columns = [smiles, target, fold, pred] + ([repeat] if repeat else [])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InputError(f"missing columns {missing}; the frame has {list(df.columns)}")
    df = df.reset_index(drop=True)
    for c in columns:
        if df[c].isna().any():
            raise InputError(f"column {c!r} is empty in rows {_rows(df[c].isna().to_numpy())}")

    task, y, classes = _resolve_task(df[target], task)
    try:
        p = pd.to_numeric(df[pred], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise InputError(f"column {pred!r} must be numeric: {exc}") from None
    if task == "binary":
        outside = (p < 0) | (p > 1)
        if outside.any():
            raise InputError(
                f"for a binary target, {pred!r} must be the probability of class {classes[1]!r}; "
                f"rows {_rows(outside)} are outside [0, 1]"
            )

    mols, mol_index = _parse_smiles(df[smiles].to_numpy())
    repeats = df[repeat] if repeat else pd.Series(0, index=df.index)
    cells = _build_cells(df[fold], repeats)
    _check_cells(cells, task, y)
    return Dataset(task=task, y=y, pred=p, mols=mols, mol_index=mol_index, cells=cells, classes=classes)
