"""A synthetic demo: para-disubstituted benzenes labelled with RDKit's Crippen logP.

Two pretend models are on offer. "noise" predicts at random, so the forest beats it. "oracle" predicts the true
label plus a little noise, so it beats the forest.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

SUBSTITUENTS = ("[H]", "C", "CC", "CCC", "O", "N", "F", "Cl", "Br", "C(=O)O", "C#N", "OC", "C(F)(F)F", "[N+](=O)[O-]")
MODELS = ("noise", "oracle")


def molecules() -> list[str]:
    """Unique canonical SMILES of every para-disubstituted benzene over SUBSTITUENTS (105 molecules)."""
    out = set()
    for a, b in itertools.combinations_with_replacement(SUBSTITUENTS, 2):
        mol = Chem.MolFromSmiles(f"c1({a})ccc({b})cc1")
        out.add(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    return sorted(out)


def demo_frame(model: str = "noise", n_folds: int = 5, n_repeats: int = 1, seed: int = 0) -> pd.DataFrame:
    """Columns smiles, target, repeat, fold, pred: ready for `forestgump.run(..., repeat="repeat")`."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    rng = np.random.default_rng(seed)
    smiles = molecules()
    y = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles])
    frames = []
    for r in range(n_repeats):
        fold = rng.permutation(np.arange(len(smiles)) % n_folds)
        noise = rng.normal(0, 0.1 if model == "oracle" else 1.5, len(smiles))
        pred = y + noise if model == "oracle" else y.mean() + noise
        frames.append(pd.DataFrame({"smiles": smiles, "target": y, "repeat": r, "fold": fold, "pred": pred}))
    return pd.concat(frames, ignore_index=True)
