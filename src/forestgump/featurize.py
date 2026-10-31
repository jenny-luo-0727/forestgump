"""Morgan fingerprints: a message-passing network with zero trainable parameters."""

from __future__ import annotations

import numpy as np
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 2048


def morgan_bits(mols, radius: int, fp_size: int = FP_SIZE) -> np.ndarray:
    """Return an (n_mols, fp_size) boolean matrix of Morgan bits."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return np.stack([gen.GetFingerprintAsNumPy(m) for m in mols]).astype(bool)


class Fingerprints:
    """Per-row Morgan bits, computed once per radius over the unique molecules.

    With several SMILES columns, each column gets its own `fp_size` bits and the blocks are concatenated in column
    order, so a substructure in one component never shares a bit with the same substructure in another."""

    def __init__(self, mols, mol_index: np.ndarray, fp_size: int = FP_SIZE):
        self._mols = mols
        self._mol_index = mol_index.reshape(len(mol_index), -1)
        self._fp_size = fp_size
        self._cache: dict[int, np.ndarray] = {}

    def __call__(self, radius: int) -> np.ndarray:
        if radius not in self._cache:
            bits = morgan_bits(self._mols, radius, self._fp_size)
            self._cache[radius] = np.hstack([bits[self._mol_index[:, j]] for j in range(self._mol_index.shape[1])])
        return self._cache[radius]
