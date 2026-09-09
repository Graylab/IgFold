"""End-to-end behaviour of fold() beyond the core model: output formats, templates, renumbering."""

import importlib.util
import os

import pytest
import torch
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from conftest import HEAVY, LIGHT, NANOBODY, requires_weights

pytestmark = requires_weights

has_anarcii = importlib.util.find_spec("anarcii") is not None
has_openmm = importlib.util.find_spec("openmm") is not None and importlib.util.find_spec("pdbfixer") is not None

FULL_H = (
    "QLQLVESGPGLVKPSQTLSLTCTVSGGSITTGYYAWSWIRQPPGKGLEWMGFIARDGSTSYSPSLKSRTSISRDTSKNQFSLQLSSVTPEDTAVYYCARAGEGRSWYPGYYYGMDYWGKGTLVTVSS"
    "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"
)


def test_no_sidecar_fasta_and_chain_numbering(runner, tmp_path):
    pdb = tmp_path / "ab.pdb"
    runner.fold(str(pdb), sequences={"H": HEAVY, "L": LIGHT}, do_refine=False, do_renum=False)
    assert not (tmp_path / "ab.fasta").exists()
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    chains = {c.id: [r.id[1] for r in c] for c in st.get_chains()}
    assert chains["H"][0] == 1 and chains["L"][0] == 1
    assert len(chains["H"]) == len(HEAVY) and len(chains["L"]) == len(LIGHT)


def test_cif_output(runner, tmp_path):
    cif = tmp_path / "nb.cif"
    out = runner.fold(str(cif), sequences={"H": NANOBODY}, do_refine=False, do_renum=False)
    assert cif.exists() and not list(tmp_path.glob("*.pdb"))
    st = MMCIFParser(QUIET=True).get_structure("x", str(cif))
    residues = list(st.get_residues())
    assert len(residues) == len(NANOBODY)
    res_rmsd = out.prmsd.square().mean(-1).sqrt()[0]
    assert residues[5]["CA"].bfactor == pytest.approx(res_rmsd[5].item(), abs=0.01)


def test_invalid_inputs_fail_before_running_model(runner, tmp_path):
    with pytest.raises(ValueError, match="non-standard"):
        runner.fold(
            str(tmp_path / "x.pdb"),
            sequences={"H": NANOBODY[:20] + "X" + NANOBODY[21:]},
            do_refine=False,
            do_renum=False,
        )
    with pytest.raises(ValueError, match="single alphanumeric"):
        runner.fold(str(tmp_path / "x.pdb"), sequences={"heavy": NANOBODY}, do_refine=False, do_renum=False)
    with pytest.raises(ValueError, match="extension"):
        runner.fold(str(tmp_path / "x.xyz"), sequences={"H": NANOBODY}, do_refine=False, do_renum=False)
    assert not list(tmp_path.iterdir())


def test_lowercase_matches_uppercase(runner, tmp_path):
    a = runner.fold(str(tmp_path / "a.pdb"), sequences={"H": NANOBODY}, do_refine=False, do_renum=False).coords
    b = runner.fold(str(tmp_path / "b.pdb"), sequences={"H": NANOBODY.lower()}, do_refine=False, do_renum=False).coords
    assert torch.allclose(a, b, atol=1e-4)


def test_template_from_own_prediction(runner, tmp_path):
    ref = tmp_path / "ref.pdb"
    ref_out = runner.fold(str(ref), sequences={"H": HEAVY, "L": LIGHT}, do_refine=False, do_renum=False)

    # same sequences, template of itself -> should stay very close to the reference
    templ_out = runner.fold(
        str(tmp_path / "t.pdb"),
        sequences={"H": HEAVY, "L": LIGHT},
        template_pdb=str(ref),
        do_refine=False,
        do_renum=False,
    )
    d = (templ_out.coords[0, :, 1] - ref_out.coords[0, :, 1]).norm(dim=-1)
    assert d.mean() < 1.0

    # a point-mutated sequence still accepts the template (alignment tolerates mismatches)
    mutated = HEAVY[:50] + "A" + HEAVY[51:]
    out = runner.fold(
        str(tmp_path / "m.pdb"),
        sequences={"H": mutated, "L": LIGHT},
        template_pdb=str(ref),
        do_refine=False,
        do_renum=False,
    )
    assert out.coords.shape[1] == len(HEAVY) + len(LIGHT)

    # ignore_chain masks a whole chain of the template
    out = runner.fold(
        str(tmp_path / "i.pdb"),
        sequences={"H": HEAVY, "L": LIGHT},
        template_pdb=str(ref),
        ignore_chain="L",
        do_refine=False,
        do_renum=False,
    )
    assert out.coords.shape[1] == len(HEAVY) + len(LIGHT)
    with pytest.raises(ValueError, match="ignore_chain"):
        runner.fold(
            str(tmp_path / "j.pdb"),
            sequences={"H": HEAVY, "L": LIGHT},
            template_pdb=str(ref),
            ignore_chain="Q",
            do_refine=False,
            do_renum=False,
        )


def test_missing_refinement_backend_error(runner, tmp_path, monkeypatch):
    from igfold.utils.folding import MissingDependencyError

    monkeypatch.setitem(os.sys.modules, "pyrosetta", None)  # make `import pyrosetta` fail
    with pytest.raises(MissingDependencyError, match="PyRosetta refinement requires"):
        runner.fold(
            str(tmp_path / "x.pdb"), sequences={"H": NANOBODY}, do_refine=True, use_openmm=False, do_renum=False
        )


@pytest.mark.skipif(not has_anarcii, reason="ANARCII not installed")
def test_renumber_and_truncate(runner, tmp_path):
    pdb = tmp_path / "renum.pdb"
    runner.fold(str(pdb), sequences={"H": HEAVY, "L": LIGHT}, do_refine=False, do_renum=True)
    st = PDBParser(QUIET=True).get_structure("x", str(pdb))
    h_ids = [r.id for r in next(c for c in st.get_chains() if c.id == "H")]
    assert h_ids[0][1] == 1 and any(ins != " " for _, _, ins in h_ids)  # Chothia insertion codes present
    lines = pdb.read_text().splitlines()
    serials = [int(l[6:11]) for l in lines if l[:3] in ("ATO", "TER")]
    assert serials == list(range(1, len(serials) + 1))

    with pytest.raises(ValueError, match="belong to the\n?.*variable domain|variable domain"):
        with pytest.warns(UserWarning):
            runner.fold(str(tmp_path / "full.pdb"), sequences={"H": FULL_H}, do_refine=False, do_renum=True)

    with pytest.warns(UserWarning, match="Truncated"):
        out = runner.fold(
            str(tmp_path / "trunc.pdb"),
            sequences={"H": FULL_H},
            do_refine=False,
            do_renum=True,
            truncate_sequences=True,
        )
    assert out.coords.shape[1] < 140


@pytest.mark.skipif(not has_openmm, reason="OpenMM/pdbfixer not installed")
def test_openmm_refine_keeps_prmsd_bfactors(runner, tmp_path):
    cif = tmp_path / "refined.cif"
    out = runner.fold(str(cif), sequences={"H": NANOBODY}, do_refine=True, use_openmm=True, do_renum=False)
    st = MMCIFParser(QUIET=True).get_structure("x", str(cif))
    residues = list(st.get_residues())
    assert len(residues) == len(NANOBODY)
    assert any(a.element == "H" for a in st.get_atoms())  # refined, all-atom
    res_rmsd = out.prmsd.square().mean(-1).sqrt()[0]
    assert residues[10]["CA"].bfactor == pytest.approx(res_rmsd[10].item(), abs=0.01)
