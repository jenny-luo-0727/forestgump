"""Per-fold metrics and the paired, corrected test that decides who won."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

METRICS = {"regression": ("rmse", "mae", "r2", "spearman"), "binary": ("roc_auc", "accuracy")}
PRIMARY = {"regression": "rmse", "binary": "roc_auc"}
LOWER_IS_BETTER = {"rmse", "mae"}
MIN_CELLS = 3


def metrics(task: str, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Score one test set. Regression reports RMSE, never MSE. Spearman is NaN when a prediction is constant."""
    if task == "binary":
        return {"roc_auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p >= 0.5))}
    spearman = np.nan if np.ptp(p) == 0 or np.ptp(y) == 0 else float(stats.spearmanr(y, p).statistic)
    return {
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "spearman": spearman,
    }


def improvement(metric: str, user: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Per-cell amount by which the user's model beats the baseline. Positive always means the user did better."""
    return baseline - user if metric in LOWER_IS_BETTER else user - baseline


@dataclass(frozen=True)
class PairedTest:
    """Nadeau-Bengio corrected resampled t on per-cell differences (positive favors the user)."""

    n_cells: int
    mean: float
    ci_low: float
    ci_high: float
    t: float
    p: float
    d_z: float


def corrected_t(d: np.ndarray, test_train_ratio: float, alpha: float = 0.05) -> PairedTest:
    """Paired t over J CV cells with the variance inflated by (1/J + n_test/n_train).

    CV training sets overlap, so the per-cell differences are positively correlated and the naive paired t is
    overconfident. Nadeau & Bengio (2003) correct the variance; Bouckaert & Frank (2004) apply it to repeated
    k-fold CV with J = k * r cells and J - 1 degrees of freedom.
    """
    d = np.asarray(d, dtype=float)
    j = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if j > 1 else 0.0
    se = np.sqrt((1 / j + test_train_ratio) * sd**2)
    if se == 0:
        t = 0.0 if mean == 0 else np.copysign(np.inf, mean)
        p = 1.0 if mean == 0 else 0.0
        half = 0.0
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), j - 1))
        half = float(stats.t.ppf(1 - alpha / 2, j - 1) * se)
    d_z = mean / sd if sd > 0 else (0.0 if mean == 0 else float(np.copysign(np.inf, mean)))
    return PairedTest(j, mean, mean - half, mean + half, float(t), p, float(d_z))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()
