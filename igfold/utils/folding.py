import os
from typing import List
from einops import rearrange
import torch
import numpy as np

from igfold.model.interface import IgFoldInput
from igfold.utils.fasta import get_fasta_chain_dict
from igfold.utils.general import exists
from igfold.utils.pdb import get_atom_coords, save_PDB, write_pdb_bfactor, cdr_indices


def get_sequence_dict(
    sequences,
    fasta_file,
):
    if exists(sequences) and exists(fasta_file):
        print("Both sequences and fasta file provided. Using fasta file.")
        seq_dict = get_fasta_chain_dict(fasta_file)
    elif not exists(sequences) and exists(fasta_file):
        seq_dict = get_fasta_chain_dict(fasta_file)
    elif exists(sequences):
        seq_dict = sequences
    else:
        exit("Must provide sequences or fasta file.")

    return seq_dict


def process_template(
    pdb_file,
    fasta_file,
    ignore_cdrs=None,
    ignore_chain=None,
):
    temp_coords, temp_mask = None, None
    if exists(pdb_file):
        temp_coords = get_atom_coords(
            pdb_file,
            fasta_file=fasta_file,
        )
        temp_coords = torch.stack(
            [
                temp_coords['N'], temp_coords['CA'], temp_coords['C'],
                temp_coords['CB']
            ],
            dim=1,
        ).view(-1, 3).unsqueeze(0)

        temp_mask = torch.ones(temp_coords.shape[:2]).bool()
        temp_mask[temp_coords.isnan().any(-1)] = False
        temp_mask[temp_coords.sum(-1) == 0] = False

        if exists(ignore_cdrs):
            cdr_names = ["h1", "h2", "h3", "l1", "l2", "l3"]
            if ignore_cdrs == False:
                cdr_names = []
            elif isinstance(ignore_cdrs, list):
                cdr_names = ignore_cdrs
            elif isinstance(ignore_cdrs, str):
                cdr_names = [ignore_cdrs]

            for cdr in cdr_names:
                cdr_range = cdr_indices(pdb_file, cdr)
                temp_mask[:, (cdr_range[0] - 1) * 4:(cdr_range[1] + 2) *
                          4] = False
        if exists(ignore_chain) and ignore_chain in ["H", "L"]:
            seq_dict = get_fasta_chain_dict(fasta_file)
            hlen = len(seq_dict["H"])
            if ignore_chain == "H":
                temp_mask[:, :hlen * 4] = False
            elif ignore_chain == "L":
                temp_mask[:, hlen * 4:] = False

    return temp_coords, temp_mask


def process_prediction(
    model_out,
    pdb_file,
    fasta_file,
    skip_pdb=False,
    do_refine=True,
    use_openmm=False,
    do_renum=False,
):
    prmsd = rearrange(
        model_out.prmsd,
        "b (l a) -> b l a",
        a=4,
    )
    model_out.prmsd = prmsd

    if skip_pdb:
        return model_out

    coords = model_out.coords.squeeze(0).detach()
    res_rmsd = prmsd.square().mean(dim=-1).sqrt().squeeze(0)

    seq_dict = get_fasta_chain_dict(fasta_file)
    full_seq = "".join(list(seq_dict.values()))
    chains = list(seq_dict.keys())
    delims = np.cumsum([len(s) for s in seq_dict.values()]).tolist()

    write_pdb = not do_refine or use_openmm
    pdb_string = save_PDB(
        pdb_file,
        coords,
        full_seq,
        chains=chains,
        atoms=['N', 'CA', 'C', 'CB', 'O'],
        error=res_rmsd,
        delim=delims,
        write_pdb=write_pdb,
    )

    if do_refine:
        if use_openmm:
            try:
                from igfold.refine.openmm_ref import refine
                refine_input = [pdb_file]
            except:
                exit("OpenMM not installed. Please install OpenMM to use refinement.")
        else:
            try:
                from igfold.refine.pyrosetta_ref import refine
                refine_input = [pdb_file, pdb_string]
            except:
                exit("PyRosetta not installed. Please install PyRosetta to use refinement.")

        refine(*refine_input)

    if do_renum:
        from igfold.utils.abnumber_ import renumber_pdb
        renumber_pdb(
            pdb_file,
            pdb_file,
        )

    write_pdb_bfactor(
        pdb_file,
        pdb_file,
        bfactor=res_rmsd,
    )

    return model_out


def fold(
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
    save_decoys=False,
    truncate_sequences=False,
):
    seq_dict = get_sequence_dict(
        sequences,
        fasta_file,
    )

    if truncate_sequences:
        from igfold.utils.abnumber_ import truncate_seq
        seq_dict = {k: truncate_seq(v) for k, v in seq_dict.items()}

    if not exists(fasta_file):
        fasta_file = pdb_file.replace(".pdb", ".fasta")
        with open(fasta_file, "w") as f:
            for chain, seq in seq_dict.items():
                f.write(">{}\n{}\n".format(
                    chain,
                    seq,
                ))

    temp_coords, temp_mask = process_template(
        template_pdb,
        fasta_file,
        ignore_cdrs=ignore_cdrs,
        ignore_chain=ignore_chain,
    )
    model_in = IgFoldInput(
        sequences=seq_dict.values(),
        template_coords=temp_coords,
        template_mask=temp_mask,
        return_embeddings=True,
    )

    model_outs, scores = [], []
    with torch.no_grad():
        for i, model in enumerate(models):
            model_out = model(model_in)
            if save_decoys:
                decoy_pdb_file = os.path.splitext(
                    pdb_file)[0] + f".decoy{i}.pdb"
                process_prediction(
                    model_out,
                    decoy_pdb_file,
                    fasta_file,
                    do_refine=do_refine,
                    use_openmm=use_openmm,
                    do_renum=do_renum,
                )

            model_out = model.gradient_refine(model_in, model_out)
            scores.append(model_out.prmsd.quantile(0.9))
            model_outs.append(model_out)

    best_model_i = scores.index(min(scores))
    model_out = model_outs[best_model_i]
    process_prediction(
        model_out,
        pdb_file,
        fasta_file,
        skip_pdb=skip_pdb,
        do_refine=do_refine,
        use_openmm=use_openmm,
        do_renum=do_renum,
    )

    return model_out
