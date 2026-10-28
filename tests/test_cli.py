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
    demo_frame("oracle").rename(columns={"pred": "my_model"}).to_csv(path, index=False)
    out = tmp_path / "forest.json"
    code = main(["run", str(path), "--pred", "my_model", "--repeat", "repeat", "--n-jobs", "2", "--json", str(out)])
    assert code == 0
    assert json.loads(out.read_text())["verdict"] == "win"


def test_bad_input_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    demo_frame("noise").to_csv(path, index=False)
    assert main(["run", str(path), "--pred", "nope"]) == EXIT_BAD_INPUT
    assert "missing columns" in capsys.readouterr().err
