import pandas as pd
import pytest

from forestgump import jokes
from forestgump.report import Comparison, Report
from forestgump.stats import PairedTest


def report(forest_delta, forest_p, mean_delta=0.0, mean_p=1.0, n_cells=5):
    tests = []
    if n_cells >= 3:
        for method, delta, p in (("mean", mean_delta, mean_p), ("knn", 0.0, 1.0), ("forest", forest_delta, forest_p)):
            tests.append(Comparison(method, PairedTest(n_cells, delta, delta - 1, delta + 1, 1.0, p, 1.0), p))
    table = pd.DataFrame({"cell": [0], "method": ["user"], "rmse": [1.0]})
    return Report(
        task="regression",
        primary="rmse",
        table=table,
        selections=[
            {"cell": 0, "method": "forest", "params": {"radius": 2, "n_estimators": 500}, "selection_score": 0.5}
        ],
        predictions=pd.DataFrame(),
        comparisons=tests,
        n_cells=n_cells,
        test_train_ratio=0.25,
        forest_seconds=41.3,
        forest_leaves=1234,
        alpha=0.05,
        seed=0,
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
    text = report(-0.5, 0.001).render(brag="fine-tuned for a week on a TPU pod")
    assert "Your model, fine-tuned for a week on a TPU pod, lost to a random forest." in text
    assert "trained in 41 seconds" in text


def test_extra_burn_for_losing_to_the_mean():
    assert jokes.LOST_TO_MEAN in report(-0.5, 0.001, mean_delta=-0.2, mean_p=0.01).render()


def test_seriously_has_no_jokes():
    text = report(-0.5, 0.001).render(seriously=True)
    assert "Verdict vs random forest: random forest better" in text
    for joke in ("feelings", "RUN, FOREST", "Forest fact", jokes.TAGLINE):
        assert joke not in text


def test_the_full_table_comes_before_the_verdict():
    text = report(-0.5, 0.001).render()
    assert text.index("Per-test-set results") < text.index("The forest won.")
