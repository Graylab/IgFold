import importlib
import os
import tempfile
import warnings
from typing import Dict, List

import numpy as np
import torch
from einops import rearrange

from igfold.model.interface import IgFoldInput
from igfold.utils.fasta import get_fasta_chain_dict
from igfold.utils.general import exists
from igfold.utils.pdb import (
    CDR_CHOTHIA_RANGES,
    LEGACY_TEMPLATE_CHAIN_MAP,
    build_structure,
    convert_structure_file,
    get_template_residues,
    output_format,
    residues_to_atom_coords,
    structure_to_pdb_string,
    write_pdb_bfactor,
    write_structure,
)

VALID_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_SEQ_LEN = 510  # AntiBERTy positional-embedding limit, minus [CLS]/[SEP]
FV_WARN_LEN = 150  # variable domains are ~100-130 residues


class MissingDependencyError(ImportError):
    """An optional dependency (PyRosetta, OpenMM, ANARCII) is not installed or not importable."""


def import_optional(module: str, feature: str, hint: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingDependencyError(
            f"{feature} requires {hint}, which could not be imported ({type(e).__name__}: {e})."
        ) from e


def get_sequence_dict(
    sequences,
    fasta_file,
) -> Dict[str, str]:
    if exists(sequences) and exists(fasta_file):
        warnings.warn("Both sequences and fasta_file were provided; using the FASTA file.")
    if exists(fasta_file):
        seq_dict = get_fasta_chain_dict(fasta_file)
    elif exists(sequences):
        seq_dict = sequences
    else:
        raise ValueError("Provide antibody sequences via `sequences` (dict of chain id -> sequence) or `fasta_file`.")

    return seq_dict


def validate_sequences(seq_dict, warn_length: bool = True) -> Dict[str, str]:
    """
    Check chain ids and sequences, returning a cleaned copy (upper-case, whitespace stripped).

    Chain ids must be single alphanumeric characters (they are written to the PDB chain column).
    Sequences must contain only the 20 standard amino acids and be at most 510 residues; a
    warning is issued for chains much longer than a variable domain.
    """
    if not isinstance(seq_dict, dict) or len(seq_dict) == 0:
        raise ValueError(
            "`sequences` must be a non-empty dict mapping chain id -> sequence, e.g. {'H': ..., 'L': ...}."
        )

    cleaned = {}
    for chain_id, seq in seq_dict.items():
        if not (isinstance(chain_id, str) and len(chain_id) == 1 and chain_id.isascii() and chain_id.isalnum()):
            raise ValueError(
                f"Chain id {chain_id!r} is invalid: chain ids must be single alphanumeric characters "
                "(e.g. 'H' and 'L'), because they are written to the PDB chain column."
            )
        if not isinstance(seq, str):
            raise ValueError(f"Sequence for chain {chain_id!r} must be a string, got {type(seq).__name__}.")

        seq = "".join(seq.split()).upper()
        if len(seq) == 0:
            raise ValueError(f"Sequence for chain {chain_id!r} is empty.")
        invalid = sorted(set(seq) - VALID_RESIDUES)
        if len(invalid) > 0:
            raise ValueError(
                f"Sequence for chain {chain_id!r} contains non-standard residues {invalid}. "
                "IgFold only supports the 20 standard amino acids."
            )
        if len(seq) > MAX_SEQ_LEN:
            raise ValueError(
                f"Chain {chain_id!r} has {len(seq)} residues, above the AntiBERTy limit of {MAX_SEQ_LEN}. "
                "IgFold predicts antibody variable domains (Fv, ~110-130 residues per chain); "
                "pass truncate_sequences=True to trim to the Fv."
            )
        if warn_length and len(seq) > FV_WARN_LEN:
            warnings.warn(
                f"Chain {chain_id!r} has {len(seq)} residues, longer than an antibody variable domain. "
                "IgFold is trained on Fv regions only and predictions for extra domains will be unreliable; "
                "pass truncate_sequences=True to trim to the Fv."
            )
        cleaned[chain_id] = seq

    if len(cleaned) > 2:
        warnings.warn(f"{len(cleaned)} chains provided; IgFold is trained on single chains and heavy/light pairs.")

    return cleaned


def truncate_sequences_to_fv(seq_dict: Dict[str, str]) -> Dict[str, str]:
    numbering = import_optional("igfold.utils.numbering", "Sequence truncation", "ANARCII (`pip install anarcii`)")

    return {k: numbering.truncate_seq(v) for k, v in seq_dict.items()}


def _cdr_names(ignore_cdrs) -> List[str]:
    if ignore_cdrs is None or ignore_cdrs is False:
        return []
    if ignore_cdrs is True:
        return list(CDR_CHOTHIA_RANGES)
    if isinstance(ignore_cdrs, str):
        ignore_cdrs = [ignore_cdrs]
    names = [c.lower() for c in ignore_cdrs]
    unknown = [c for c in names if c not in CDR_CHOTHIA_RANGES]
    if len(unknown) > 0:
        raise ValueError(f"Unknown CDR names {unknown}; expected any of {list(CDR_CHOTHIA_RANGES)}.")

    return names


def check_template_args(template_pdb, ignore_cdrs, ignore_chain, seq_dict: Dict[str, str]):
    """Validate template options before any model work is done."""
    cdrs = _cdr_names(ignore_cdrs)
    if exists(ignore_chain) and ignore_chain not in seq_dict:
        raise ValueError(f"ignore_chain={ignore_chain!r} is not one of the sequence chains {list(seq_dict)}.")
    if (len(cdrs) > 0 or exists(ignore_chain)) and not exists(template_pdb):
        raise ValueError("ignore_cdrs and ignore_chain only apply when template_pdb is given.")
    if exists(template_pdb) and not os.path.isfile(template_pdb):
        raise FileNotFoundError(f"Template structure not found: {template_pdb}")


def process_template(
    template_pdb,
    seq_dict: Dict[str, str],
    ignore_cdrs=None,
    ignore_chain=None,
):
    """
    Extract backbone template coordinates (1, 4L, 3) and mask (1, 4L) aligned to the input
    sequences. Template residues are matched to the sequences by alignment, so templates may
    have missing residues or point mutations.

    :param ignore_cdrs: CDR names (Chothia-numbered template with chains H/L) to mask out,
        True for all six, or None.
    :param ignore_chain: chain id whose template coordinates should be masked out.
    """
    if not exists(template_pdb):
        return None, None

    residues, keys = get_template_residues(template_pdb, seq_dict)
    atom_coords = residues_to_atom_coords(residues)
    temp_coords = (
        torch.stack(
            [atom_coords["N"], atom_coords["CA"], atom_coords["C"], atom_coords["CB"]],
            dim=1,
        )
        .view(-1, 3)
        .unsqueeze(0)
    )

    res_mask = torch.tensor([exists(r) for r in residues])
    n_res = len(residues)

    for cdr in _cdr_names(ignore_cdrs):
        chain_letter = cdr[0].upper()
        lo, hi = CDR_CHOTHIA_RANGES[cdr]
        in_cdr = [
            exists(r)
            and LEGACY_TEMPLATE_CHAIN_MAP.get(r.get_parent().id, r.get_parent().id) == chain_letter
            and lo <= r.id[1] <= hi
            for r in residues
        ]
        for i, flag in enumerate(in_cdr):
            if flag:
                # mask the loop plus one flanking residue on each side (within the same chain)
                for j in (i - 1, i, i + 1):
                    if 0 <= j < n_res and keys[j] == keys[i]:
                        res_mask[j] = False

    if exists(ignore_chain):
        if ignore_chain not in seq_dict:
            raise ValueError(f"ignore_chain={ignore_chain!r} is not one of the sequence chains {list(seq_dict)}.")
        for i, k in enumerate(keys):
            if k == ignore_chain:
                res_mask[i] = False

    temp_mask = res_mask.repeat_interleave(4).unsqueeze(0)
    temp_mask[temp_coords.isnan().any(-1)] = False
    temp_mask[temp_coords.sum(-1) == 0] = False
    temp_coords = torch.nan_to_num(temp_coords)

    return temp_coords, temp_mask


def process_prediction(
    model_out,
    out_file,
    seq_dict: Dict[str, str],
    skip_pdb=False,
    do_refine=True,
    use_openmm=False,
    do_renum=False,
    log=None,
):
    """Write (and optionally refine and renumber) a prediction; ``log`` is an optional message callback."""
    prmsd = model_out.prmsd
    if prmsd.dim() == 2:
        prmsd = rearrange(prmsd, "b (l a) -> b l a", a=4)
    model_out.prmsd = prmsd

    if skip_pdb:
        return model_out

    coords = model_out.coords.squeeze(0).detach()
    res_rmsd = prmsd.square().mean(dim=-1).sqrt().squeeze(0)

    full_seq = "".join(seq_dict.values())
    chains = list(seq_dict.keys())
    delims = np.cumsum([len(s) for s in seq_dict.values()]).tolist()

    structure = build_structure(coords, full_seq, chains, delims, bfactor=res_rmsd)

    # Refinement and renumbering tools work on PDB files. Work on a temporary file whenever the
    # output is not the plain unrefined PDB, so a failure never leaves a partial result behind.
    fmt = output_format(out_file)
    out_dir = os.path.dirname(os.path.abspath(out_file))
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "pdb" and not do_refine and not do_renum:
        work_pdb = out_file
    else:
        fd, work_pdb = tempfile.mkstemp(suffix=".pdb", prefix="igfold_", dir=out_dir)
        os.close(fd)

    try:
        if do_refine:
            if use_openmm:
                openmm_ref = import_optional(
                    "igfold.refine.openmm_ref",
                    "OpenMM refinement",
                    "OpenMM and pdbfixer (`conda install -c conda-forge openmm pdbfixer`)",
                )
                write_structure(structure, work_pdb)
                openmm_ref.refine(work_pdb)
            else:
                pyrosetta_ref = import_optional(
                    "igfold.refine.pyrosetta_ref",
                    "PyRosetta refinement",
                    "PyRosetta (see http://pyrosetta.org/downloads), or pass use_openmm=True",
                )
                pyrosetta_ref.refine(work_pdb, structure_to_pdb_string(structure))
        else:
            write_structure(structure, work_pdb)

        if do_renum:
            numbering = import_optional("igfold.utils.numbering", "Renumbering", "ANARCII (`pip install anarcii`)")
            numbering.renumber_pdb(work_pdb, work_pdb)
            if exists(log):
                log("Chothia renumbering complete.")

        if do_refine:
            # refined files carry no B-factors; write the predicted RMSD
            write_pdb_bfactor(work_pdb, work_pdb, bfactor=res_rmsd)

        if work_pdb != out_file:
            if fmt == "pdb":
                os.replace(work_pdb, out_file)
            else:
                convert_structure_file(work_pdb, out_file)
    finally:
        if work_pdb != out_file and os.path.exists(work_pdb):
            os.remove(work_pdb)

    return model_out


def fold(
    antiberty,
    models,
    pdb_file,
    fasta_file=None,
    sequences=None,
    template_pdb=None,
    ignore_cdrs=None,
    ignore_chain=None,
    skip_pdb=False,
    do_refine=True,
    use_openmm=False,
    do_renum=True,
    truncate_sequences=False,
    log=None,
):
    seq_dict = get_sequence_dict(sequences, fasta_file)
    seq_dict = validate_sequences(seq_dict, warn_length=not truncate_sequences)

    if truncate_sequences:
        seq_dict = validate_sequences(truncate_sequences_to_fv(seq_dict))

    check_template_args(template_pdb, ignore_cdrs, ignore_chain, seq_dict)
    if not skip_pdb:
        output_format(pdb_file)  # fail early on an unsupported extension

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

    model_outs, scores = [], []
    with torch.no_grad():
        for model in models:
            model_out = model(model_in)
            model_out = model.gradient_refine(model_in, model_out)
            scores.append(model_out.prmsd.quantile(0.9))
            model_outs.append(model_out)

    best_model_i = scores.index(min(scores))
    model_out = model_outs[best_model_i]
    process_prediction(
        model_out,
        pdb_file,
        seq_dict,
        skip_pdb=skip_pdb,
        do_refine=do_refine,
        use_openmm=use_openmm,
        do_renum=do_renum,
        log=log,
    )

    return model_out


# ---------------------------------------------------------------------------
# Batched prediction
# ---------------------------------------------------------------------------


def _slice_output(model_out, i: int, res_idx: torch.Tensor):
    """Extract sample ``i`` (real residues ``res_idx``) from a padded batched IgFoldOutput."""
    from igfold.model.interface import IgFoldOutput

    def res(t):
        return None if t is None else t[i : i + 1, res_idx]

    prmsd = rearrange(model_out.prmsd, "b (l a) -> b l a", a=4)[i : i + 1, res_idx]

    return IgFoldOutput(
        coords=res(model_out.coords),
        prmsd=rearrange(prmsd, "b l a -> b (l a)"),
        translations=res(model_out.translations),
        rotations=res(model_out.rotations),
        bert_embs=res(model_out.bert_embs),
        bert_attn=None if model_out.bert_attn is None else model_out.bert_attn[i : i + 1][:, res_idx][:, :, res_idx],
        gt_embs=res(model_out.gt_embs),
        structure_embs=res(model_out.structure_embs),
    )


def _pad_chain_features(embs: List[List[torch.Tensor]], atts: List[List[torch.Tensor]]):
    """
    Pad per-sample, per-chain AntiBERTy features to a batch. ``embs[i][c]`` is (1, l, 512) and
    ``atts[i][c]`` is (1, layers, heads, l, l). Returns per-chain padded embeddings/attentions,
    the residue mask (B, L) and the per-sample real residue indices.
    """
    n_samples, n_chains = len(embs), len(embs[0])
    max_lens = [max(embs[i][c].shape[1] for i in range(n_samples)) for c in range(n_chains)]

    embeddings, attentions, chain_masks = [], [], []
    for c in range(n_chains):
        e0, a0 = embs[0][c], atts[0][c]
        e = e0.new_zeros(n_samples, max_lens[c], e0.shape[-1])
        a = a0.new_zeros(n_samples, a0.shape[1], a0.shape[2], max_lens[c], max_lens[c])
        m = torch.zeros(n_samples, max_lens[c], dtype=torch.bool, device=e0.device)
        for i in range(n_samples):
            l = embs[i][c].shape[1]
            e[i, :l] = embs[i][c][0]
            a[i, :, :, :l, :l] = atts[i][c][0]
            m[i, :l] = True
        embeddings.append(e)
        attentions.append(a)
        chain_masks.append(m)

    res_mask = torch.cat(chain_masks, dim=1)
    res_idx = [res_mask[i].nonzero(as_tuple=True)[0] for i in range(n_samples)]

    return embeddings, attentions, res_mask, res_idx


def predict_structures(
    antiberty,
    models,
    seq_dicts: List[Dict[str, str]],
    batch_size: int = 8,
) -> List:
    """
    Predict structures for many antibodies, batching the network forward pass. Sequences with
    the same set of chain ids are padded into batches of at most ``batch_size``; the per-sample
    gradient refinement and model selection are then identical to :func:`fold`.

    :param seq_dicts: sequence dicts (validated here; see :func:`validate_sequences`).
    :return: one IgFoldOutput per input, in order (``coords`` (1, L, 5, 3), ``prmsd`` (1, L, 4)).
    """
    seq_dicts = [validate_sequences(s, warn_length=False) for s in seq_dicts]
    results = [None] * len(seq_dicts)

    # group by chain layout, preserving first-seen order
    groups: Dict[tuple, List[int]] = {}
    for i, s in enumerate(seq_dicts):
        groups.setdefault(tuple(s.keys()), []).append(i)

    for chain_ids, indices in groups.items():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_seqs = [seq_dicts[i] for i in batch_idx]
            n_chains = len(chain_ids)

            flat = [seq for s in batch_seqs for seq in s.values()]
            flat_embs, flat_atts = antiberty.embed(flat, return_attention=True)
            embs = [
                [flat_embs[i * n_chains + c][1:-1].unsqueeze(0) for c in range(n_chains)]
                for i in range(len(batch_seqs))
            ]
            atts = [
                [flat_atts[i * n_chains + c][:, :, 1:-1, 1:-1].unsqueeze(0) for c in range(n_chains)]
                for i in range(len(batch_seqs))
            ]

            embeddings, attentions, res_mask, res_idx = _pad_chain_features(embs, atts)
            batch_in = IgFoldInput(
                embeddings=embeddings,
                attentions=attentions,
                batch_mask=res_mask.repeat_interleave(4, dim=1),
                return_embeddings=True,
            )

            best = [(None, None)] * len(batch_seqs)  # (score, output) per sample
            with torch.no_grad():
                for model in models:
                    batch_out = model(batch_in)
                    for j in range(len(batch_seqs)):
                        sample_in = IgFoldInput(embeddings=embs[j], attentions=atts[j], return_embeddings=True)
                        sample_out = model.gradient_refine(sample_in, _slice_output(batch_out, j, res_idx[j]))
                        score = sample_out.prmsd.quantile(0.9)
                        if best[j][0] is None or score < best[j][0]:
                            best[j] = (score, sample_out)

            for j, i in enumerate(batch_idx):
                out = best[j][1]
                out.prmsd = rearrange(out.prmsd, "b (l a) -> b l a", a=4)
                results[i] = out

    return results


def fold_batch(
    antiberty,
    models,
    out_files: List[str],
    sequences: List[Dict[str, str]],
    batch_size: int = 8,
    do_refine=True,
    use_openmm=False,
    do_renum=True,
    truncate_sequences=False,
    log=None,
):
    """Predict and write structures for many antibodies; see :func:`predict_structures`."""
    if len(out_files) != len(sequences):
        raise ValueError(f"Got {len(out_files)} output files for {len(sequences)} sequence sets.")

    seq_dicts = [validate_sequences(s, warn_length=not truncate_sequences) for s in sequences]
    if truncate_sequences:
        seq_dicts = [validate_sequences(truncate_sequences_to_fv(s)) for s in seq_dicts]
    for f in out_files:
        output_format(f)

    outputs = predict_structures(antiberty, models, seq_dicts, batch_size=batch_size)
    for out, out_file, seq_dict in zip(outputs, out_files, seq_dicts):
        process_prediction(
            out,
            out_file,
            seq_dict,
            do_refine=do_refine,
            use_openmm=use_openmm,
            do_renum=do_renum,
            log=log,
        )

    return outputs
