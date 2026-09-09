import numpy as np
import pytest
import torch
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

from igfold.utils.pdb import (
    align_residues_to_sequence,
    build_structure,
    fix_atom_serials,
    match_template_chains,
    output_format,
    structure_to_pdb_string,
    write_structure,
)


def _fake_coords(n):
    # straight-ish chain with distinct atom positions
    base = torch.arange(n, dtype=torch.float32)[:, None] * torch.tensor([3.8, 0.0, 0.0])
    offsets = torch.tensor([[0, 0, 0], [1.4, 0.5, 0], [2.5, 0, 0.3], [1.4, 1.9, 0.4], [3.0, -1.0, 0.5]])
    return base[:, None, :] + offsets[None]


def test_output_format():
    assert output_format("x.pdb") == "pdb"
    assert output_format("x.CIF") == "cif"
    assert output_format("x.mmcif") == "cif"
    with pytest.raises(ValueError):
        output_format("x.xyz")


def test_pdb_numbering_restarts_per_chain_and_serials_unique(tmp_path):
    seq = "GAVLI" + "STCM"
    st = build_structure(_fake_coords(9), seq, ["H", "L"], [5, 9], bfactor=torch.arange(9.0))
    path = str(tmp_path / "out.pdb")
    write_structure(st, path)
    lines = open(path).read().splitlines()
    atoms = [l for l in lines if l.startswith("ATOM")]
    ters = [l for l in lines if l.startswith("TER")]

    assert len(ters) == 2
    assert [l[21] for l in atoms].count("H") == 5 * 5 - 1  # GLY has no CB
    assert int([l for l in atoms if l[21] == "L"][0][22:26]) == 1
    serials = [int(l[6:11]) for l in lines if l[:3] in ("ATO", "TER")]
    assert serials == list(range(1, len(serials) + 1))
    assert float(atoms[-1][60:66]) == 8.0  # B-factor carries the per-residue value


def test_structure_roundtrip_via_biopython(tmp_path):
    seq = "GAVLISTCM"
    st = build_structure(_fake_coords(9), seq, ["A"], [9])
    path = str(tmp_path / "out.pdb")
    write_structure(st, path)
    parsed = PDBParser(QUIET=True).get_structure("x", path)
    residues = list(parsed.get_residues())
    assert len(residues) == 9
    np.testing.assert_allclose(residues[1]["CA"].coord, _fake_coords(9)[1, 1].numpy(), atol=1e-3)
    assert structure_to_pdb_string(st).startswith("ATOM")


def test_cif_output(tmp_path):
    st = build_structure(_fake_coords(4), "GAVL", ["H"], [4], bfactor=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    path = str(tmp_path / "out.cif")
    write_structure(st, path)
    parsed = MMCIFParser(QUIET=True).get_structure("x", path)
    residues = list(parsed.get_residues())
    assert [r.resname for r in residues] == ["GLY", "ALA", "VAL", "LEU"]
    assert residues[3]["CA"].bfactor == pytest.approx(4.0)


def test_build_structure_rejects_bad_inputs():
    with pytest.raises(ValueError, match="not a standard amino acid"):
        build_structure(_fake_coords(2), "AX", ["H"], [2])
    with pytest.raises(ValueError, match="single alphanumeric"):
        build_structure(_fake_coords(2), "AA", ["heavy"], [2])
    with pytest.raises(ValueError, match="delimiters"):
        build_structure(_fake_coords(2), "AA", ["H"], [1])


def test_fix_atom_serials(tmp_path):
    path = tmp_path / "dup.pdb"
    path.write_text(
        "ATOM      1  N   ALA H   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "TER       2      ALA H   1\n"
        "ATOM      2  N   ALA L   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "CONECT    1    2\n"
        "END\n"
    )
    fix_atom_serials(str(path))
    lines = path.read_text().splitlines()
    assert [int(l[6:11]) for l in lines[:3]] == [1, 2, 3]
    assert lines[3].split() == ["CONECT", "1", "3"]


def test_align_with_mutation_and_gap(tmp_path):
    seq = "GAVLISTCMKR"
    st = build_structure(_fake_coords(11), seq, ["H"], [11])
    residues = list(st.get_residues())
    # template missing two N-terminal residues, one point mutation (S->A)
    template_res = residues[2:]
    target = "GAVLIATCMKR"
    mapped, identity = align_residues_to_sequence(template_res, target)
    assert mapped[0] is None and mapped[1] is None
    assert mapped[2] is residues[2] and mapped[-1] is residues[-1]
    assert identity == pytest.approx(8 / 9)


def test_match_template_chains():
    assert match_template_chains(["H", "L"], {"H": "A", "L": "A"}) == {"H": "H", "L": "L"}
    assert match_template_chains(["A", "B"], {"H": "A", "L": "A"}) == {"H": "A", "L": "B"}
    assert match_template_chains(["X"], {"N": "A"}) == {"N": "X"}
    with pytest.raises(ValueError, match="Could not match"):
        match_template_chains(["X", "Y"], {"H": "A", "L": "A"})
