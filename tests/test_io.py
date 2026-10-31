import numpy as np
import pandas as pd
import pytest

from forestgump import InputError, load
from forestgump.demo import demo_frame


def frame(**overrides):
    df = demo_frame("noise").drop(columns="repeat")
    for column, (row, value) in overrides.items():
        df[column] = df[column].astype(object)
        df.loc[row, column] = value
    return df


def test_cells_partition_each_repeat_and_never_overlap():
    df = demo_frame("noise", n_repeats=2)
    ds = load(df, repeat="repeat")
    assert len(ds.cells) == 10
    for r in (0, 1):
        cells = [c for c in ds.cells if c.repeat == r]
        tests = np.concatenate([c.test_idx for c in cells])
        assert sorted(tests) == list(np.flatnonzero(df["repeat"] == r))
        for c in cells:
            assert not set(c.train_idx) & set(c.test_idx)
            assert set(c.train_idx) | set(c.test_idx) == set(np.flatnonzero(df["repeat"] == r))


def test_missing_column():
    with pytest.raises(InputError, match="missing columns"):
        load(frame(), pred="my_model")


def test_context_columns_must_exist_and_be_filled():
    with pytest.raises(InputError, match=r"missing columns \['series'\]"):
        load(frame(), context="series")
    df = frame().assign(series="a", batch="x")
    df.loc[3, "series"] = None
    with pytest.raises(InputError, match="'series' is empty in rows 3"):
        load(df, context=["batch", "series"])


def test_context_groups_combine_the_columns():
    df = frame().assign(a=np.arange(105) % 2, b=np.arange(105) % 3)
    ds = load(df, context=["a", "b"])
    assert ds.context_columns == ("a", "b") and len(np.unique(ds.context)) == 6
    assert load(df, context="a").context_columns == ("a",) and load(df).context is None


def test_bad_smiles_names_the_row():
    with pytest.raises(InputError, match="rows 7"):
        load(frame(smiles=(7, "c1ccc")))


def test_empty_prediction_names_the_row():
    with pytest.raises(InputError, match="'pred' is empty in rows 3"):
        load(frame(pred=(3, np.nan)))


def test_single_fold_has_no_training_set():
    df = frame()
    df["fold"] = 0
    with pytest.raises(InputError, match="single fold"):
        load(df)


def test_binary_is_inferred_and_needs_probabilities():
    df = frame()
    df["target"] = np.where(df["target"] > df["target"].median(), "high", "low")
    df["pred"] = 0.5
    ds = load(df)
    assert ds.task == "binary" and ds.classes == ("high", "low")
    assert set(np.unique(ds.y)) == {0.0, 1.0}
    df.loc[0, "pred"] = 1.5
    with pytest.raises(InputError, match="probability of class 'low'"):
        load(df)


def test_tiny_training_set_is_refused():
    df = pd.concat([frame().head(6)], ignore_index=True)
    df["fold"] = [0, 1, 2, 0, 1, 2]
    with pytest.raises(InputError, match="not a shrub"):
        load(df)


def test_several_smiles_columns_are_parsed_per_column():
    df = frame().assign(solvent=np.where(np.arange(105) % 2, "O", "CCO"))
    ds = load(df, smiles=["smiles", "solvent"])
    assert ds.smiles_columns == ("smiles", "solvent") and ds.mol_index.shape == (105, 2)
    assert {ds.mols[i].GetNumAtoms() for i in ds.mol_index[:, 1]} == {1, 3}
    with pytest.raises(InputError, match=r"missing columns \['base'\]"):
        load(df, smiles=["smiles", "base"])
    df.loc[4, "solvent"] = "C1CC"
    with pytest.raises(InputError, match="rows 4"):
        load(df, smiles=["smiles", "solvent"])
