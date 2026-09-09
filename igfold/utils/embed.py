from einops import rearrange

from igfold.model.interface import IgFoldInput
from igfold.utils.folding import check_template_args, get_sequence_dict, process_template, validate_sequences


def embed(
    antiberty,
    model,
    fasta_file=None,
    sequences=None,
    template_pdb=None,
    ignore_cdrs=None,
    ignore_chain=None,
):
    seq_dict = validate_sequences(get_sequence_dict(sequences, fasta_file))
    check_template_args(template_pdb, ignore_cdrs, ignore_chain, seq_dict)

    embeddings, attentions = antiberty.embed(
        list(seq_dict.values()),
        return_attention=True,
    )
    embeddings = [e[1:-1].unsqueeze(0) for e in embeddings]
    attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in attentions]

    temp_coords, temp_mask = process_template(
        template_pdb,
        seq_dict,
        ignore_cdrs=ignore_cdrs,
        ignore_chain=ignore_chain,
    )
    model_in = IgFoldInput(
        embeddings=embeddings,
        attentions=attentions,
        template_coords=temp_coords,
        template_mask=temp_mask,
        return_embeddings=True,
    )

    model_out = model(model_in)

    prmsd = rearrange(
        model_out.prmsd,
        "b (l a) -> b l a",
        a=4,
    )
    model_out.prmsd = prmsd

    return model_out


def pool_embeddings(model_out, seq_dict, reduce: str = "mean"):
    """
    Fixed-size (sequence-level) embeddings from an :func:`embed` output.

    :param reduce: "mean" or "max" over residues.
    :return: dict keyed by chain id plus "all" (all chains together); each value is a dict with
        ``bert_embs`` (512,), ``gt_embs`` (64,) and ``structure_embs`` (64,) tensors.
    """
    if reduce not in ("mean", "max"):
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}.")

    def pool(t):
        return t.mean(dim=0) if reduce == "mean" else t.max(dim=0).values

    feats = {
        "bert_embs": model_out.bert_embs[0],
        "gt_embs": model_out.gt_embs[0],
        "structure_embs": model_out.structure_embs[0],
    }
    pooled, start = {}, 0
    for chain_id, seq in seq_dict.items():
        end = start + len(seq)
        pooled[chain_id] = {k: pool(v[start:end]) for k, v in feats.items()}
        start = end
    pooled["all"] = {k: pool(v) for k, v in feats.items()}

    return pooled
