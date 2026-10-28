# forestgump

*Life is like a box of baselines.*

**Morgan-fingerprint baselines for cheminformatics models.**

You trained a molecular property model for 3 days on 8×H100. Did it beat a random forest trained on Morgan
fingerprints?

forestgump takes your molecules as SMILES, plus your labels, your cross-validation folds and your model's
out-of-fold predictions. It fits three baselines on **exactly the same folds**: the training mean, Tanimoto kNN on
Morgan fingerprints, and a random forest on Morgan fingerprints. Then it runs a paired test to decide whether your
model beat the forest.

```
Your model, trained on 8×H100 for 3 days, lost to a random forest. The forest trained in 31 seconds on a CPU.
The forest has no GPU. The forest has no feelings. The forest won.
```

## Install

```bash
pip install forestgump
```

## Try it

```bash
forestgump demo                             # a model that predicts noise. The forest wins.
forestgump demo --model oracle              # knows the answer, beats the forest on all 5 folds, and only ties
forestgump demo --model oracle --repeats 3  # 15 folds is enough evidence. You win.
```

The demo molecules are 105 para-disubstituted benzenes. Their labels are RDKit's Crippen logP.

The tie is not a bug. Five folds is not much evidence, and the test is built not to call a winner without it.

## Use it

You need one CSV with one row per molecule. Its columns are:

| column | meaning |
|---|---|
| `smiles` | the molecule |
| `target` | the label: a number, or one of two classes |
| `fold` | which test fold the row belongs to |
| `pred` | your model's prediction for the row, made while it was in the test fold. For a two-class target, this is the probability of the larger class label. |

```bash
forestgump run my_results.csv
forestgump run my_results.csv --smiles SMILES --target y --fold cv_fold --pred my_model
```

For **repeated** cross-validation, add one row per (molecule, repeat) and pass `--repeat repeat_column`. The test
set of (repeat r, fold f) is those rows. Its training set is every other row of repeat r.

From Python:

```python
import forestgump

report = forestgump.run(df, smiles="smiles", target="target", fold="fold", pred="pred")
print(report.verdict)  # "win", "lose", "tie", or "refuse"
print(report.render())
report.to_json("forest.json")
```

## The baselines

Each baseline is retrained on every training set, and its hyperparameters are picked from **that training set
only**. Test rows are only ever predicted.

- **Large Lookup Model** uses inner cross-validation, which takes milliseconds for kNN: 5 folds, or 3 below 50
  training rows.
- **The Forest** fits each of its 12 configs once and keeps the one with the best out-of-bag score. Every tree
  skips about a third of the training rows, so the forest gets a validation score for free. If you'd rather tune
  it like everything else, `--forest-inner-cv` uses inner cross-validation instead. That fits each config once
  per inner fold and then refits the winner, so it is about 5× slower.

| name | what it is | grid |
|---|---|---|
| The Mean | the training-set mean, or the class prior | none (zero parameters, zero shots, zero GPUs) |
| Large Lookup Model | k-nearest neighbors with Tanimoto distance | k ∈ {1, 3, 5, 10} × {uniform, distance} weights × radius ∈ {2, 3} |
| The Forest | a random forest with 500 trees | max_features ∈ {sqrt, 0.3, 1.0} × min_samples_leaf ∈ {1, 3} × radius ∈ {2, 3} |

Both fingerprint baselines use 2048-bit Morgan fingerprints from RDKit.

## How the verdict is decided

The primary metric is RMSE for numeric targets and ROC-AUC for two-class targets. The report also gives MAE, R²
and Spearman ρ, or accuracy.

1. Every method is scored on every test set. The full per-test-set table is printed **before** any verdict.
2. Your model is compared with each baseline on the per-test-set differences, using the **Nadeau–Bengio corrected
   resampled t-test**. CV training sets overlap, so the differences are correlated and a plain paired t-test is
   overconfident. The correction inflates the variance by `1/J + n_test/n_train` over J test sets
   (Nadeau & Bengio 2003; Bouckaert & Frank 2004 for repeated CV).
3. The three p-values are Holm-adjusted.
4. The verdict compares you with the forest at α = 0.05: **win**, **lose**, or **tie**. With fewer than 3 test
   sets the forest declines to be judged.

The report also gives the mean difference, its confidence interval and Cohen's d_z. Use `--json` to get every
number, the hyperparameters chosen on each test set, and the baselines' predictions.

## Flags

| flag | effect |
|---|---|
| `--seriously` | plain output with no jokes, for logs and papers |
| `--brag "fine-tuned for a week on a TPU pod"` | changes how the forest describes your training |
| `--forest-inner-cv` | picks the forest's hyperparameters by inner CV instead of out-of-bag score. About 5× slower, and it warns you. |
| `--fail-if-forest-wins` | exits 1 if the forest won, so a CI pipeline goes red when your model loses |
| `--task regression\|binary` | overrides the task. By default a target with 2 values is binary. |
| `--seed`, `--n-jobs`, `--alpha` | what you'd expect |

## License

Apache-2.0.
