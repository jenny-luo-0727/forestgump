import numpy as np
import pandas as pd
import pytest

import forestgump
from forestgump.demo import demo_frame

POISON = 1e6


@pytest.mark.filterwarnings("ignore:forest_selection='inner-cv'")
@pytest.mark.parametrize("forest_selection", ["oob", "inner-cv"])
def test_test_rows_never_reach_selection_or_fitting(small_forest, forest_selection):
    """Poison one fold's labels. That fold's test set must not see them; every other fold trains on them."""
    df = demo_frame("noise").drop(columns="repeat")
    df["series"] = np.arange(len(df)) % 4
    df.loc[df["fold"] == 0, "target"] = POISON
    report = forestgump.run(df, context="series", n_jobs=2, forest_selection=forest_selection)
    clean = report.table.query("method != 'user'").loc[lambda t: t["fold"] == 0, "cell"].unique()
    assert len(clean) == 1
    cell = clean[0]

    preds = report.predictions
    assert preds.loc[preds["cell"] == cell, "pred"].abs().max() < 1e3
    chosen = [s["selection_score"] for s in report.selections if s["cell"] == cell and s["method"] in ("knn", "forest")]
    assert all(np.isfinite(chosen)) and max(chosen) < 1e3
    # The poison is visible to every other cell's training set, so the test can fail.
    for method in ("mean", "context"):
        assert preds.loc[(preds["cell"] != cell) & (preds["method"] == method), "pred"].min() > 1e4


def test_context_mean_is_each_groups_training_mean(small_forest):
    """Series 9 exists only in fold 0, so fold 0's test rows fall back to the training mean."""
    df = demo_frame("noise").drop(columns="repeat")
    df["series"] = np.where(df["fold"] == 0, 9, np.arange(len(df)) % 4)
    report = forestgump.run(df, context="series", n_jobs=2)
    folds = report.table.drop_duplicates("cell").set_index("cell")["fold"]
    preds = report.predictions.query("method == 'context'")
    for cell, f in folds.items():
        train = df[df["fold"] != f]
        means = train.groupby("series")["target"].mean()
        test = preds[preds["cell"] == cell]
        expected = df.loc[test["row"], "series"].map(means).fillna(train["target"].mean())
        np.testing.assert_allclose(test["pred"], expected.to_numpy())
    unseen = {s["cell"]: s["n_test_unseen_context"] for s in report.selections if s["method"] == "context"}
    n_fold0 = int((df["fold"] == 0).sum())
    assert unseen == {cell: n_fold0 if f == 0 else 0 for cell, f in folds.items()}
    assert len(report.comparisons) == 4
    assert f"{n_fold0} of {len(df)} test rows had no such training row" in report.render()


def test_selection_searches_the_full_grid(small_forest):
    report = forestgump.run(demo_frame("noise"), repeat="repeat", n_jobs=2)
    knn = [s for s in report.selections if s["method"] == "knn"]
    assert len(knn) == 5
    assert all(s["params"]["n_neighbors"] in (1, 3, 5, 10) for s in knn)


def test_binary_runs(small_forest):
    df = demo_frame("oracle")
    median = df["target"].median()
    df["pred"] = 1 / (1 + np.exp(-(df["pred"] - median) * 5))
    df["target"] = (df["target"] > median).astype(int)
    df["series"] = np.arange(len(df)) % 3
    report = forestgump.run(df, repeat="repeat", context="series", n_jobs=2)
    assert report.task == "binary" and report.primary == "roc_auc"
    assert set(report.table.columns) >= {"roc_auc", "accuracy"}
    assert isinstance(report.table, pd.DataFrame) and len(report.table) == 5 * 5
    context = report.predictions.query("method == 'context'")["pred"]
    assert context.between(0, 1).all() and context.nunique() > 1


def test_forest_inner_cv_warns_and_is_reported(small_forest):
    with pytest.warns(UserWarning, match="about 5x slower"):
        report = forestgump.run(demo_frame("noise"), repeat="repeat", n_jobs=2, forest_selection="inner-cv")
    assert report.to_dict()["forest_selection"] == "inner-cv"
    text = report.render()
    assert "the forest by inner CV" in text and "Forest hyperparameters chosen by inner CV" in text


def test_unknown_forest_selection_is_refused():
    with pytest.raises(ValueError, match="forest_selection"):
        forestgump.run(demo_frame("noise"), repeat="repeat", forest_selection="vibes")
