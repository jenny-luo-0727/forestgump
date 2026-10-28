import numpy as np
import pytest
from scipy import stats

from forestgump.stats import corrected_t, holm, improvement, metrics


def test_corrected_t_matches_hand_computation():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = corrected_t(d, test_train_ratio=0.25)
    # mean 3, sample variance 2.5, corrected variance (1/5 + 0.25) * 2.5 = 1.125
    assert out.mean == 3.0
    assert out.t == pytest.approx(3 / np.sqrt(1.125))
    assert out.p == pytest.approx(2 * stats.t.sf(3 / np.sqrt(1.125), df=4))
    half = stats.t.ppf(0.975, df=4) * np.sqrt(1.125)
    assert (out.ci_low, out.ci_high) == pytest.approx((3 - half, 3 + half))
    assert out.d_z == pytest.approx(3 / np.sqrt(2.5))


def test_correction_is_wider_than_the_naive_paired_t():
    d = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
    naive = stats.ttest_1samp(d, 0).pvalue
    assert corrected_t(d, 0.25).p > naive
    assert corrected_t(d, 0.0).p == pytest.approx(naive)


def test_zero_variance():
    assert corrected_t(np.zeros(4), 0.25).p == 1.0
    assert corrected_t(np.full(4, 0.2), 0.25).p == 0.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_improvement_is_positive_when_the_user_is_better():
    assert improvement("rmse", np.array([1.0]), np.array([2.0]))[0] == 1.0
    assert improvement("roc_auc", np.array([0.9]), np.array([0.7]))[0] == pytest.approx(0.2)


def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = metrics("regression", y, y + np.array([1.0, -1.0, 1.0, -1.0]))
    assert m["rmse"] == 1.0 and m["mae"] == 1.0
    assert "mse" not in m
    assert np.isnan(metrics("regression", y, np.full(4, 2.5))["spearman"])
