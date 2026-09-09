import torch
from conftest import HEAVY, LIGHT, NANOBODY, requires_weights

pytestmark = requires_weights


def _ca_ca_distances(coords):
    ca = coords[0, :, 1]  # atoms: N, CA, C, CB, O
    return (ca[1:] - ca[:-1]).norm(dim=-1)


def test_fold_nanobody(runner, tmp_path):
    pdb = tmp_path / "nb.pdb"
    out = runner.fold(str(pdb), sequences={"H": NANOBODY}, do_refine=False, do_renum=False)

    assert out.coords.shape == (1, len(NANOBODY), 5, 3)
    assert out.prmsd.shape == (1, len(NANOBODY), 4)
    assert torch.isfinite(out.coords).all()
    # sequential CA-CA distances should be close to the 3.8 A of a trans peptide
    d = _ca_ca_distances(out.coords)
    assert (d > 2.8).all() and (d < 4.5).all()
    assert pdb.exists() and pdb.read_text().count("\nATOM") > 500


def test_fold_paired(runner, tmp_path):
    pdb = tmp_path / "ab.pdb"
    out = runner.fold(str(pdb), sequences={"H": HEAVY, "L": LIGHT}, do_refine=False, do_renum=False)

    n = len(HEAVY) + len(LIGHT)
    assert out.coords.shape == (1, n, 5, 3)
    # heavy and light chains must be in contact (Fv interface), not flung apart
    ca = out.coords[0, :, 1]
    min_interchain = torch.cdist(ca[: len(HEAVY)], ca[len(HEAVY) :]).min()
    assert min_interchain < 6.0
    text = pdb.read_text()
    assert text.count("\nTER") == 2 and " H " in text and " L " in text


def test_embed(runner):
    emb = runner.embed(sequences={"H": HEAVY, "L": LIGHT})
    n = len(HEAVY) + len(LIGHT)
    assert emb.bert_embs.shape == (1, n, 512)
    assert emb.gt_embs.shape == (1, n, 64)
    assert emb.structure_embs.shape == (1, n, 64)


def test_fold_is_deterministic(runner, tmp_path):
    kw = dict(sequences={"H": NANOBODY}, do_refine=False, do_renum=False)
    a = runner.fold(str(tmp_path / "a.pdb"), **kw).coords
    b = runner.fold(str(tmp_path / "b.pdb"), **kw).coords
    assert torch.allclose(a, b, atol=1e-4)
