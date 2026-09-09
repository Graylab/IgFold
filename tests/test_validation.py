import warnings

import pytest
from conftest import HEAVY, LIGHT

from igfold.utils.folding import MissingDependencyError, get_sequence_dict, import_optional, validate_sequences


def test_cleans_case_and_whitespace():
    out = validate_sequences({"H": " evql\nvq "})
    assert out == {"H": "EVQLVQ"}


def test_rejects_nonstandard_residue():
    with pytest.raises(ValueError, match="non-standard residues \\['X'\\]"):
        validate_sequences({"H": "EVQLX"})


def test_rejects_multichar_chain_id():
    with pytest.raises(ValueError, match="single alphanumeric"):
        validate_sequences({"heavy": HEAVY})


def test_rejects_empty_and_non_dict():
    with pytest.raises(ValueError):
        validate_sequences({})
    with pytest.raises(ValueError):
        validate_sequences([HEAVY])
    with pytest.raises(ValueError, match="empty"):
        validate_sequences({"H": ""})


def test_too_long_is_error_and_long_is_warning():
    with pytest.raises(ValueError, match="truncate_sequences=True"):
        validate_sequences({"H": "A" * 511})
    with pytest.warns(UserWarning, match="longer than an antibody variable domain"):
        validate_sequences({"H": "A" * 200})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_sequences({"H": "A" * 200}, warn_length=False)
        validate_sequences({"H": HEAVY, "L": LIGHT})


def test_get_sequence_dict_requires_input():
    with pytest.raises(ValueError, match="sequences"):
        get_sequence_dict(None, None)


def test_get_sequence_dict_from_fasta(tmp_path):
    fasta = tmp_path / "ab.fasta"
    fasta.write_text(f">1abc:H\n{HEAVY}\n>L\n{LIGHT}\n")
    assert get_sequence_dict(None, str(fasta)) == {"H": HEAVY, "L": LIGHT}


def test_import_optional_error_type():
    with pytest.raises(MissingDependencyError, match="Frobnication requires"):
        import_optional("igfold_not_a_module", "Frobnication", "the frob package")
    assert issubclass(MissingDependencyError, ImportError)
