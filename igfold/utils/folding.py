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
            elif type(ignore_cdrs) == List:
                cdr_names = ignore_cdrs
            elif type(ignore_cdrs) == str:
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
    use_abnum=False,
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
    delims = np.cumsum([len(s) for s in seq_dict.values()]).tolist()
    save_PDB(
        pdb_file,
        coords,
        full_seq,
        atoms=['N', 'CA', 'C', 'CB', 'O'],
        error=res_rmsd,
        delim=delims,
    )

    if do_refine:
        if use_openmm:
            from igfold.refine.openmm_ref import refine
        else:
            try:
                from igfold.refine.pyrosetta_ref import refine
            except ImportError as e:
                print(
                    "Warning: PyRosetta not available. Using OpenMM instead.")
                print(e)
                from igfold.refine.openmm_ref import refine

        refine(pdb_file)

    if do_renum:
        if use_abnum:
            from igfold.utils.pdb import renumber_pdb
        else:
            try:
                from igfold.utils.anarci_ import renumber_pdb
            except ImportError as e:
                print(
                    "Warning: ANARCI not available. Provide --use_abnum to renumber with the AbNum server."
                )
                print(e)
                renumber_pdb = lambda x, y: None

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
    use_abnum=False,
    save_decoys=False,
):
    seq_dict = get_sequence_dict(
        sequences,
        fasta_file,
    )
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
                    use_abnum=use_abnum,
                )

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
        use_abnum=use_abnum,
    )

    return model_out
