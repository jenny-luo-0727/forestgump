import pytest

from forestgump import baselines


@pytest.fixture
def small_forest(monkeypatch):
    """Shrink the forest so tests run in seconds. The logic under test does not depend on its size.

    The tree count travels inside each grid config, so worker processes see it too.
    """
    monkeypatch.setattr(
        baselines,
        "GRIDS",
        baselines.GRIDS
        | {
            "forest": [
                {"radius": 2, "n_estimators": 25, "max_features": "sqrt", "min_samples_leaf": 1},
                {"radius": 2, "n_estimators": 25, "max_features": 1.0, "min_samples_leaf": 3},
            ]
        },
    )
