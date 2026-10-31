import numpy as np

from forestgump import load
from forestgump.demo import demo_frame
from forestgump.featurize import FP_SIZE, Fingerprints, morgan_bits


def test_one_column_gives_the_plain_morgan_bits():
    ds = load(demo_frame("noise").drop(columns="repeat"))
    assert np.array_equal(Fingerprints(ds.mols, ds.mol_index)(2), morgan_bits(ds.mols, 2)[ds.mol_index[:, 0]])


def test_several_columns_concatenate_one_block_each():
    df = demo_frame("noise").drop(columns="repeat").assign(solvent=np.where(np.arange(105) % 2, "O", "CCO"))
    ds = load(df, smiles=["smiles", "solvent"])
    X = Fingerprints(ds.mols, ds.mol_index)(2)
    assert X.shape == (105, 2 * FP_SIZE)
    bits = morgan_bits(ds.mols, 2)
    assert np.array_equal(X[:, :FP_SIZE], bits[ds.mol_index[:, 0]])
    assert np.array_equal(X[:, FP_SIZE:], bits[ds.mol_index[:, 1]])
