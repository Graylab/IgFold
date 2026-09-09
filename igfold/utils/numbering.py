"""
Antibody numbering and variable-domain truncation with ANARCII.
"""

import contextlib
import io
import warnings
from typing import List, Tuple

from anarcii import Anarcii
from Bio.PDB import PDBIO, PDBParser
from Bio.SeqUtils import seq1

from igfold.utils.pdb import fix_atom_serials

_MODELS = {}

Position = Tuple[int, str]  # residue number, insertion code (" " if none)


def _model(cpu: bool = True):
    key = ("antibody", cpu)
    if key not in _MODELS:
        _MODELS[key] = Anarcii(seq_type="antibody", mode="accuracy", cpu=cpu, verbose=False)

    return _MODELS[key]


def number_sequences(seqs: List[str], scheme: str = "chothia", min_score: float = 0.0):
    """
    Number antibody variable domains.

    :return: one entry per sequence: ``(numbering, start, end)`` where ``numbering`` is a list of
        ``((number, insertion), residue)`` for the numbered residues and ``seq[start:end]`` is the
        numbered region (the variable domain).
    :raises ValueError: if a sequence cannot be numbered.
    """
    model = _model()
    with contextlib.redirect_stdout(io.StringIO()):  # ANARCII prints progress even when not verbose
        model.number(list(seqs))
        results = list(model.to_scheme(scheme).values())

    out = []
    for seq, r in zip(seqs, results):
        if r["error"] is not None or r["numbering"] is None or r["score"] < min_score:
            raise ValueError(
                f"ANARCII could not number the sequence {seq[:20]}... ({r['error'] or 'low score'}). "
                "IgFold expects one antibody variable domain (Fv) per chain."
            )
        numbering = [((int(num), ins if ins.strip() else " "), aa) for (num, ins), aa in r["numbering"] if aa != "-"]
        start, end = int(r["query_start"]), int(r["query_end"]) + 1
        if "".join(aa for _, aa in numbering) != seq[start:end]:
            raise RuntimeError("ANARCII numbering does not match the input sequence.")
        out.append((numbering, start, end))

    return out


def truncate_seq(seq: str, scheme: str = "chothia") -> str:
    """Trim a sequence to its numbered variable domain."""
    _, start, end = number_sequences([seq], scheme=scheme)[0]
    truncated = seq[start:end]
    if len(truncated) < len(seq):
        warnings.warn(f"Truncated sequence from {len(seq)} to {len(truncated)} residues (variable domain).")

    return truncated


MAX_UNNUMBERED = 15  # residues outside the variable domain tolerated per chain when renumbering


def renumber_pdb(
    in_pdb_file,
    out_pdb_file=None,
    scheme: str = "chothia",
):
    """
    Renumber the residues of a predicted structure (default Chothia scheme). A few residues
    outside the numbered variable domain are allowed and keep sequential numbers (counting
    down before the first numbered residue, up after the last); longer extensions raise.
    """
    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    structure = PDBParser(QUIET=True).get_structure("_", in_pdb_file)[0]
    chains = list(structure.get_chains())
    chain_residues = [[r for r in chain.get_residues() if r.id[0] == " "] for chain in chains]
    seqs = ["".join(seq1(r.resname) for r in residues) for residues in chain_residues]

    for chain, residues, seq, (numbering, start, end) in zip(
        chains, chain_residues, seqs, number_sequences(seqs, scheme)
    ):
        n_extra = start + (len(seq) - end)
        if n_extra > MAX_UNNUMBERED:
            raise ValueError(
                f"Chain {chain.id} has {len(seq)} residues but only residues {start + 1}-{end} belong to the "
                "variable domain: predict the Fv only, or pass truncate_sequences=True."
            )
        if n_extra > 0:
            warnings.warn(
                f"Chain {chain.id}: {n_extra} residue(s) outside the variable domain kept with sequential numbers."
            )

        new_ids = []
        first_num, last_num = numbering[0][0][0], numbering[-1][0][0]
        for i in range(start):
            new_ids.append((" ", first_num - (start - i), " "))
        for residue, ((num, ins), aa) in zip(residues[start:end], numbering):
            if seq1(residue.resname) != aa:
                raise RuntimeError(f"Residue mismatch while renumbering chain {chain.id} of {in_pdb_file}.")
            new_ids.append((" ", num, ins))
        for i in range(len(seq) - end):
            new_ids.append((" ", last_num + i + 1, " "))

        # assign in two passes so no two residues share an id at any point
        for residue in residues:
            residue.id = (" ", -10000 - residues.index(residue), " ")
        for residue, new_id in zip(residues, new_ids):
            residue.id = new_id

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)
    fix_atom_serials(out_pdb_file)
