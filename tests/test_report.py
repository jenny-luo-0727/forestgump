import pandas as pd
import pytest

from forestgump import jokes
from forestgump.report import Comparison, Report
from forestgump.stats import PairedTest


def report(forest_delta, forest_p, mean_delta=0.0, mean_p=1.0, n_cells=5, knn_delta=0.0, knn_p=1.0, context_delta=None):
    tests = []
    baselines = [("mean", mean_delta, mean_p), ("knn", knn_delta, knn_p), ("forest", forest_delta, forest_p)]
    if context_delta is not None:
        baselines.insert(1, ("context", context_delta, 0.01))
    if n_cells >= 3:
        for method, delta, p in baselines:
            tests.append(Comparison(method, PairedTest(n_cells, delta, delta - 1, delta + 1, 1.0, p, 1.0), p))
    table = pd.DataFrame({"cell": [0], "method": ["user"], "n_test": [21], "rmse": [1.0]})
    selections = [{"cell": 0, "method": "forest", "params": {"radius": 2, "n_estimators": 500}, "selection_score": 0.5}]
    if context_delta is not None:
        selections.append({"cell": 0, "method": "context", "params": {}, "n_test_unseen_context": 4})
    return Report(
        task="regression",
        primary="rmse",
        table=table,
        selections=selections,
        predictions=pd.DataFrame(),
        comparisons=tests,
        n_cells=n_cells,
        test_train_ratio=0.25,
        forest_seconds=41.3,
        forest_leaves=1234,
        alpha=0.05,
        seed=0,
        context=("ligand", "base") if context_delta is not None else (),
    )


@pytest.mark.parametrize(
    ("delta", "p", "verdict", "phrase"),
    [
        (-0.5, 0.001, "lose", "The forest won."),
        (0.5, 0.001, "win", "cannot clap"),
        (0.5, 0.2, "tie", "unimpressed"),
        (-0.5, 0.2, "tie", "unimpressed"),
    ],
)
def test_verdicts(delta, p, verdict, phrase):
    r = report(delta, p)
    assert r.verdict == verdict
    assert phrase in r.render()


def test_refusal_below_three_cells():
    r = report(0, 1, n_cells=2)
    assert r.verdict == "refuse"
    assert "declines to be judged on n=2" in r.render()


def test_brag_and_seconds_in_the_losing_line():
    text = report(-0.5, 0.001).render(brag="a 3-billion-parameter foundation model")
    assert "Your model, a 3-billion-parameter foundation model, lost to a random forest." in text
    assert "trained in 41 seconds" in text


def test_extra_burn_for_losing_to_the_mean():
    assert jokes.LOST_TO_MEAN in report(-0.5, 0.001, mean_delta=-0.2, mean_p=0.01).render()


def test_context_mean_is_described_tested_and_can_burn():
    text = report(0.5, 0.001, context_delta=-0.2).render()
    assert "sharing the test row's ligand, base. 4 of 21 test rows had no such training row" in text
    assert "Holm across the 4 baselines" in text and "Retrieval-Augmented Mean (context mean)" in text
    assert "cannot clap" in text and jokes.LOST_TO_CONTEXT in text
    assert report(0.5, 0.001, context_delta=0.2).to_dict()["context_columns"] == ["ligand", "base"]
    plain = report(0.5, 0.001)
    assert "Holm across the 3 baselines" in plain.render() and "context mean" not in plain.render().lower()


def test_seriously_has_no_jokes():
    text = report(-0.5, 0.001).render(seriously=True)
    assert "Verdict vs random forest: random forest better" in text
    for joke in ("feelings", "RUN, FOREST", "Forest fact", jokes.TAGLINE):
        assert joke not in text


def test_the_full_table_comes_before_the_verdict():
    text = report(-0.5, 0.001).render()
    assert text.index("Per-test-set results") < text.index("The forest won.")


def test_beating_the_forest_but_losing_to_knn():
    text = report(0.5, 0.001, knn_delta=-0.2, knn_p=0.01).render()
    assert "cannot clap" in text and jokes.LOST_TO_KNN in text
