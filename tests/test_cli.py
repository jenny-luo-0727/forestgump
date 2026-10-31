import json

from forestgump.cli import EXIT_BAD_INPUT, EXIT_FOREST_WON, main
from forestgump.demo import demo_frame


def test_demo_noise_loses_and_ci_fails(small_forest, tmp_path, capsys):
    out = tmp_path / "forest.json"
    assert main(["demo", "--n-jobs", "2", "--json", str(out)]) == 0
    assert "The forest won." in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["verdict"] == "lose" and result["n_test_sets"] == 5
    assert main(["demo", "--n-jobs", "2", "--fail-if-forest-wins", "--seriously"]) == EXIT_FOREST_WON


def test_run_on_a_csv(small_forest, tmp_path, capsys):
    path = tmp_path / "oracle.csv"
    df = demo_frame("oracle").rename(columns={"pred": "my_model"})
    df.assign(series=df.index % 3, batch="b").to_csv(path, index=False)
    out = tmp_path / "forest.json"
    argv = ["run", str(path), "--pred", "my_model", "--repeat", "repeat", "--context", "series", "batch"]
    code = main([*argv, "--n-jobs", "2", "--json", str(out)])
    assert code == 0
    result = json.loads(out.read_text())
    assert result["verdict"] == "win" and result["context_columns"] == ["series", "batch"]
    assert [t["baseline"] for t in result["paired_tests"]] == ["mean", "context", "knn", "forest"]


def test_bad_input_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    demo_frame("noise").to_csv(path, index=False)
    assert main(["run", str(path), "--pred", "nope"]) == EXIT_BAD_INPUT
    assert "missing columns" in capsys.readouterr().err


def test_forest_inner_cv_prints_one_clean_warning(small_forest, recwarn, capsys):
    assert main(["demo", "--n-jobs", "2", "--forest-inner-cv", "--seriously"]) == 0
    err = capsys.readouterr().err
    assert err.count("about 5x slower") == 1 and "UserWarning" not in err
    assert not [w for w in recwarn if "inner-cv" in str(w.message)]


def test_run_with_several_smiles_columns(small_forest, tmp_path, capsys):
    path, out = tmp_path / "components.csv", tmp_path / "forest.json"
    df = demo_frame("noise").drop(columns="repeat")
    df.assign(solvent=["O" if i % 2 else "CCO" for i in range(len(df))]).to_csv(path, index=False)
    assert (
        main(["run", str(path), "--smiles", "smiles", "solvent", "--seriously", "--n-jobs", "2", "--json", str(out)])
        == 0
    )
    assert "one 2048-bit Morgan block per SMILES column (smiles, solvent)" in capsys.readouterr().out
    assert json.loads(out.read_text())["smiles_columns"] == ["smiles", "solvent"]
