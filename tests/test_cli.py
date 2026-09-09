import pytest
from conftest import NANOBODY, requires_weights

from igfold.cli import build_parser, main


def test_parser_requires_command_and_output():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fold", "-H", "EVQL"])


def test_fold_requires_sequences(tmp_path):
    with pytest.raises(SystemExit, match="provide sequences"):
        main(["fold", "-o", str(tmp_path / "x.pdb")])


def test_runner_import_paths():
    from igfold import IgFoldRunner
    from igfold.runner import IgFoldRunner as runner_cls

    assert IgFoldRunner is runner_cls
    # the old submodule name is gone on purpose: it shadowed the class when imported directly
    with pytest.raises(ImportError):
        import igfold.IgFoldRunner  # noqa: F401


@requires_weights
def test_cli_fold(tmp_path, capsys):
    out = tmp_path / "nb.cif"
    rc = main(["fold", "-H", NANOBODY, "-o", str(out), "--num-models", "1", "--device", "cpu"])
    assert rc == 0 and out.exists()
    assert "median predicted RMSD" in capsys.readouterr().out
