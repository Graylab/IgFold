import pytest
import torch
from conftest import HEAVY, LIGHT, NANOBODY, requires_weights

pytestmark = requires_weights

NB2 = "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNSKNTVYLQMNSLRPEDTAVYYCAAGRYYSDYWGQGTQVTVSS"
L2 = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGQGTKVEIK"


def test_padded_forward_matches_single(runner):
    """The network forward pass on a padded batch must reproduce single-sample outputs."""
    from igfold.model.interface import IgFoldInput
    from igfold.utils.folding import _pad_chain_features, _slice_output

    model = runner.models[0]
    seqs = [{"H": HEAVY, "L": LIGHT}, {"H": HEAVY[:100], "L": L2}]
    flat = [s for d in seqs for s in d.values()]
    fe, fa = runner.antiberty.embed(flat, return_attention=True)
    embs = [[fe[i * 2 + c][1:-1].unsqueeze(0) for c in range(2)] for i in range(2)]
    atts = [[fa[i * 2 + c][:, :, 1:-1, 1:-1].unsqueeze(0) for c in range(2)] for i in range(2)]
    E, A, res_mask, res_idx = _pad_chain_features(embs, atts)
    assert not res_mask.all()  # second sample is padded

    with torch.no_grad():
        batched = model(
            IgFoldInput(
                embeddings=E, attentions=A, batch_mask=res_mask.repeat_interleave(4, dim=1), return_embeddings=True
            )
        )
        for j in range(2):
            single = model(IgFoldInput(embeddings=embs[j], attentions=atts[j], return_embeddings=True))
            b = _slice_output(batched, j, res_idx[j])
            # O atoms are excluded: their placement depends on chain ends, which the padded
            # forward pass cannot know; gradient_refine re-places them with the true chain lengths
            assert (b.coords[:, :, :4] - single.coords[:, :, :4]).norm(dim=-1).max() < 1e-3
            assert (b.prmsd - single.prmsd).abs().max() < 1e-3


def test_batched_matches_single(runner):
    seqs = [{"H": HEAVY, "L": LIGHT}, {"H": HEAVY, "L": L2}, {"H": NB2}, {"H": NANOBODY}]
    batched = runner.predict(seqs, batch_size=8)
    assert len(batched) == 4
    for s, out in zip(seqs, batched):
        n = sum(len(v) for v in s.values())
        assert out.coords.shape == (1, n, 5, 3) and out.prmsd.shape == (1, n, 4)
        single = runner.fold("unused.pdb", sequences=s, skip_pdb=True, do_refine=False, do_renum=False)
        # the 80-step gradient refinement amplifies float noise from the padded forward pass
        assert (out.coords - single.coords).norm(dim=-1).max() < 0.25
        assert (out.prmsd - single.prmsd).abs().max() < 0.1


def test_batch_size_chunking_and_order(runner):
    seqs = [{"H": NANOBODY}, {"H": NB2}, {"H": NANOBODY[:-3]}]
    a = runner.predict(seqs, batch_size=1)
    b = runner.predict(seqs, batch_size=3)
    for x, y in zip(a, b):
        assert x.coords.shape == y.coords.shape
        assert (x.coords - y.coords).norm(dim=-1).max() < 0.25
    assert [o.coords.shape[1] for o in a] == [len(NANOBODY), len(NB2), len(NANOBODY) - 3]


def test_fold_batch_writes_files(runner, tmp_path):
    files = [str(tmp_path / "a.pdb"), str(tmp_path / "b.cif")]
    outs = runner.fold_batch(files, [{"H": NANOBODY}, {"H": HEAVY, "L": LIGHT}], do_refine=False, do_renum=False)
    assert len(outs) == 2 and all((tmp_path / f).exists() for f in ("a.pdb", "b.cif"))
    with pytest.raises(ValueError, match="output files"):
        runner.fold_batch(files, [{"H": NANOBODY}], do_refine=False, do_renum=False)


def test_sequence_embedding(runner):
    pooled = runner.sequence_embedding(sequences={"H": HEAVY, "L": LIGHT})
    assert set(pooled) == {"H", "L", "all"}
    assert pooled["H"]["bert_embs"].shape == (512,)
    assert pooled["L"]["gt_embs"].shape == (64,) and pooled["all"]["structure_embs"].shape == (64,)
    full = runner.embed(sequences={"H": HEAVY, "L": LIGHT})
    assert torch.allclose(pooled["H"]["bert_embs"], full.bert_embs[0, : len(HEAVY)].mean(0))
    mx = runner.sequence_embedding(sequences={"H": HEAVY}, reduce="max")
    heavy_only = runner.embed(sequences={"H": HEAVY})  # chains interact, so embed H alone for comparison
    assert torch.allclose(mx["all"]["gt_embs"], heavy_only.gt_embs[0].max(0).values, atol=1e-4)
    with pytest.raises(ValueError):
        runner.sequence_embedding(sequences={"H": HEAVY}, reduce="median")
