"""The report: the full per-cell grid first, then the paired tests, then the verdict."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import jokes
from .stats import LOWER_IS_BETTER, METRICS, PairedTest

GOOFY_NAMES = {
    "user": "Your model",
    "mean": "The Mean",
    "knn": "Large Lookup Model (Tanimoto kNN)",
    "forest": "The Forest (Morgan RF)",
}
SERIOUS_NAMES = {"user": "user model", "mean": "mean", "knn": "kNN (Tanimoto)", "forest": "random forest (Morgan)"}


@dataclass(frozen=True)
class Comparison:
    method: str
    test: PairedTest
    p_holm: float


@dataclass(frozen=True)
class Report:
    task: str
    primary: str
    table: pd.DataFrame
    selections: list[dict]
    predictions: pd.DataFrame
    comparisons: list[Comparison]
    n_cells: int
    test_train_ratio: float
    forest_seconds: float
    forest_leaves: int
    alpha: float
    seed: int
    classes: tuple | None = None

    def _comparison(self, method: str) -> Comparison | None:
        return next((c for c in self.comparisons if c.method == method), None)

    def _outcome(self, method: str) -> str:
        c = self._comparison(method)
        if c is None:
            return "refuse"
        if c.p_holm < self.alpha:
            return "win" if c.test.mean > 0 else "lose"
        return "tie"

    @property
    def verdict(self) -> str:
        """'win', 'lose' or 'tie' for the user's model against the forest; 'refuse' below 3 cells."""
        return self._outcome("forest")

    def render(self, *, seriously: bool = False, brag: str = jokes.DEFAULT_BRAG) -> str:
        names = SERIOUS_NAMES if seriously else GOOFY_NAMES
        direction = "lower" if self.primary in LOWER_IS_BETTER else "higher"
        out = [] if seriously else [jokes.BANNER]
        out.append(
            f"Design: {self.n_cells} paired test sets (repeat x fold); every method is scored on the same rows. "
            f"Baseline hyperparameters are chosen per test set from its training rows only "
            f"(the forest by out-of-bag score, kNN by inner CV). "
            f"Primary metric: {self.primary} ({direction} is better)."
        )

        table = self.table.assign(method=self.table["method"].map(names))
        out += ["", "Per-test-set results", table.to_string(index=False, float_format=lambda v: f"{v:.4g}")]

        if self.comparisons:
            out += [
                "",
                f"Paired tests on {self.primary}: Nadeau-Bengio corrected t over {self.n_cells} test sets "
                f"(mean n_test/n_train = {self.test_train_ratio:.3f}), Holm across the 3 baselines.",
                f"delta > 0 means your model did better. CI is the unadjusted {1 - self.alpha:.0%} interval.",
                self._tests_frame(names).to_string(index=False, float_format=lambda v: f"{v:.4g}"),
            ]

        out += ["", self._verdict_text(seriously, brag)]
        out += ["", self._forest_configs()]
        if not seriously:
            n_trees = next(s["params"]["n_estimators"] for s in self.selections if s["method"] == "forest")
            out += ["", jokes.MODEL_CARD.format(n_trees=n_trees, n_leaves=self.forest_leaves)]
            out += ["", jokes.fact(self.seed), "", jokes.TAGLINE]
        return "\n".join(out)

    def _tests_frame(self, names: dict) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "baseline": names[c.method],
                "delta": c.test.mean,
                "ci_low": c.test.ci_low,
                "ci_high": c.test.ci_high,
                "t": c.test.t,
                "p": c.test.p,
                "p_holm": c.p_holm,
                "d_z": c.test.d_z,
            }
            for c in self.comparisons
        )

    def _verdict_text(self, seriously: bool, brag: str) -> str:
        if self.verdict == "refuse":
            if seriously:
                return f"Verdict: none. {self.n_cells} test set(s) is too few for a paired test (need 3)."
            return jokes.refuse(self.n_cells)
        forest = self._comparison("forest")
        if seriously:
            word = {"win": "user model better", "lose": "random forest better", "tie": "no significant difference"}
            return f"Verdict vs random forest: {word[self.verdict]} (Holm-adjusted p = {forest.p_holm:.3g})."
        if self.verdict == "lose":
            lines = [jokes.lose(brag, self.forest_seconds)]
        elif self.verdict == "win":
            lines = [jokes.win(forest.p_holm)]
        else:
            lines = [jokes.tie(forest.p_holm)]
        if self._outcome("mean") == "lose":
            lines.append(jokes.LOST_TO_MEAN)
        if self._outcome("knn") == "lose":
            lines.append(jokes.LOST_TO_KNN)
        return "\n".join(lines)

    def _forest_configs(self) -> str:
        picks = Counter(
            ", ".join(f"{k}={v}" for k, v in s["params"].items()) for s in self.selections if s["method"] == "forest"
        )
        lines = [f"Forest hyperparameters chosen by out-of-bag score (of {self.n_cells} test sets):"]
        lines += [f"  {n} x  {cfg}" for cfg, n in picks.most_common()]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Every number in the report, JSON-safe (NaN becomes null)."""
        from . import __version__

        return _jsonable(
            {
                "forestgump_version": __version__,
                "task": self.task,
                "classes": list(self.classes) if self.classes else None,
                "primary_metric": self.primary,
                "metrics": list(METRICS[self.task]),
                "n_test_sets": self.n_cells,
                "mean_test_train_ratio": self.test_train_ratio,
                "alpha": self.alpha,
                "seed": self.seed,
                "verdict": self.verdict,
                "forest_seconds": self.forest_seconds,
                "per_test_set": self.table.to_dict(orient="records"),
                "paired_tests": [
                    {"baseline": c.method, "p_holm": c.p_holm} | c.test.__dict__ for c in self.comparisons
                ],
                "selections": self.selections,
                "baseline_predictions": self.predictions.to_dict(orient="records"),
            }
        )

    def to_json(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
