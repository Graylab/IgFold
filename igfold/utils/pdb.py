import warnings
from bisect import bisect_left, bisect_right
from os.path import basename, splitext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBIO, PDBParser
from Bio.SeqUtils import seq1

from igfold.utils.coordinates import place_fourth_atom
from igfold.utils.general import _aa_1_3_dict, exists

# Chothia CDR definitions (inclusive residue-number ranges)
CDR_CHOTHIA_RANGES = {
    "h1": (26, 32),
    "h2": (52, 56),
    "h3": (95, 102),
    "l1": (24, 34),
    "l2": (50, 56),
    "l3": (89, 97),
}

# Alternative template chain ids accepted for the heavy / light chain
LEGACY_TEMPLATE_CHAIN_MAP = {"A": "H", "B": "L"}


def _parse_structure(pdb_file):
    """First model of a PDB file as a Bio.PDB Model."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure(splitext(basename(pdb_file))[0], pdb_file)[0]


def get_atom_coord(residue, atom_type):
    if exists(residue) and atom_type in residue:
        return residue[atom_type].get_coord()
    else:
        return [0, 0, 0]


def get_cb_or_ca_coord(residue):
    if not exists(residue):
        return [0, 0, 0]

    if "CB" in residue:
        return residue["CB"].get_coord()
    elif "CA" in residue:
        return residue["CA"].get_coord()
    else:
        return [0, 0, 0]


def get_atom_coords_mask(coords):
    mask = torch.tensor([1 if sum(_) != 0 else 0 for _ in coords], dtype=torch.uint8)
    mask = mask & (1 - torch.any(torch.isnan(coords), dim=1).to(torch.uint8))
    return mask


def place_missing_cb_o(atom_coords):
    cb_coords = place_fourth_atom(
        atom_coords["C"],
        atom_coords["N"],
        atom_coords["CA"],
        torch.tensor(1.522),
        torch.tensor(1.927),
        torch.tensor(-2.143),
    )
    o_coords = place_fourth_atom(
        torch.roll(atom_coords["N"], shifts=-1, dims=0),
        atom_coords["CA"],
        atom_coords["C"],
        torch.tensor(1.231),
        torch.tensor(2.108),
        torch.tensor(-3.142),
    )

    bb_mask = (
        get_atom_coords_mask(atom_coords["N"])
        & get_atom_coords_mask(atom_coords["CA"])
        & get_atom_coords_mask(atom_coords["C"])
    )
    missing_cb = (get_atom_coords_mask(atom_coords["CB"]) & bb_mask) == 0
    atom_coords["CB"][missing_cb] = cb_coords[missing_cb]

    bb_mask = (
        get_atom_coords_mask(
            torch.roll(
                atom_coords["N"],
                shifts=-1,
                dims=0,
            )
        )
        & get_atom_coords_mask(atom_coords["CA"])
        & get_atom_coords_mask(atom_coords["C"])
    )
    missing_o = (get_atom_coords_mask(atom_coords["O"]) & bb_mask) == 0
    atom_coords["O"][missing_o] = o_coords[missing_o]


def align_residues_to_sequence(
    residues: List,
    target_seq: str,
) -> Tuple[List, float]:
    """
    Map the residues of a PDB chain onto positions of ``target_seq`` by global sequence
    alignment. Mismatches (e.g. point mutations) are tolerated; unaligned positions are None.

    :return: list of length ``len(target_seq)`` with a Bio.PDB residue or None per position,
        and the fraction of aligned target positions that are identical.
    """
    pdb_seq = "".join(seq1(r.get_resname()) for r in residues)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5
    # Missing termini in the template are common and should be cheap
    try:  # Biopython >= 1.86 naming
        aligner.end_insertion_score = 0
        aligner.end_deletion_score = 0
    except AttributeError:
        aligner.target_end_gap_score = 0
        aligner.query_end_gap_score = 0

    alignment = aligner.align(target_seq, pdb_seq)[0]

    mapped = [None for _ in target_seq]
    n_aligned, n_identical = 0, 0
    for (t_start, t_end), (q_start, q_end) in zip(*alignment.aligned):
        for i in range(t_end - t_start):
            mapped[t_start + i] = residues[q_start + i]
            n_aligned += 1
            n_identical += target_seq[t_start + i] == pdb_seq[q_start + i]

    identity = n_identical / n_aligned if n_aligned > 0 else 0.0

    return mapped, identity


def match_template_chains(
    template_chain_ids: List[str],
    seq_dict: Dict[str, str],
) -> Dict[str, str]:
    """
    Decide which template chain provides coordinates for each sequence key.
    Matches identical ids first, then the legacy A->H / B->L convention, and finally
    pairs a single template chain with a single sequence.
    """
    unused = list(template_chain_ids)
    mapping = {}
    for key in seq_dict:
        if key in unused:
            mapping[key] = key
            unused.remove(key)
    for tid in list(unused):
        key = LEGACY_TEMPLATE_CHAIN_MAP.get(tid)
        if exists(key) and key in seq_dict and key not in mapping:
            mapping[key] = tid
            unused.remove(tid)
    if len(mapping) == 0 and len(unused) == 1 and len(seq_dict) == 1:
        mapping[next(iter(seq_dict))] = unused[0]

    if len(mapping) == 0:
        raise ValueError(
            f"Could not match template chains {template_chain_ids} to sequence chains {list(seq_dict)}. "
            "Name the template chains to match the sequence dictionary keys (e.g. H and L)."
        )

    return mapping


def get_template_residues(
    pdb_file: str,
    seq_dict: Dict[str, str],
    min_identity: float = 0.5,
) -> Tuple[List, List[str]]:
    """
    Return one Bio.PDB residue (or None) per position of the concatenated sequences, taken
    from the template structure, together with the sequence key each position belongs to.
    """
    structure = _parse_structure(pdb_file)
    chains = {c.id: c for c in structure.get_chains()}
    mapping = match_template_chains(list(chains), seq_dict)

    residues, keys = [], []
    for key, seq in seq_dict.items():
        if key in mapping:
            chain_residues = [r for r in chains[mapping[key]].get_residues() if r.id[0] == " "]
            mapped, identity = align_residues_to_sequence(chain_residues, seq)
            if identity < min_identity:
                warnings.warn(
                    f"Template chain {mapping[key]} aligns to sequence {key} with only "
                    f"{identity:.0%} identity; check that the right template was provided."
                )
        else:
            warnings.warn(f"No template chain found for sequence {key}; it will be predicted without a template.")
            mapped = [None for _ in seq]

        residues += mapped
        keys += [key] * len(seq)

    return residues, keys


def get_atom_coords(pdb_file: str, seq_dict: Optional[Dict[str, str]] = None):
    """
    Backbone (N, CA, C, CB, O) coordinates from a PDB file, ordered by the concatenated
    sequences in ``seq_dict`` when given (missing positions are zero), else by file order.
    """
    if exists(seq_dict):
        residues, _ = get_template_residues(pdb_file, seq_dict)
    else:
        residues = list(_parse_structure(pdb_file).get_residues())

    return residues_to_atom_coords(residues)


def residues_to_atom_coords(residues: List):
    """Backbone (N, CA, C, CB, O) coordinate tensors for a list of Bio.PDB residues (None -> zeros)."""
    atom_coords = {}
    for atom in ["N", "CA", "C", "CB", "O"]:
        atom_coords[atom] = torch.tensor(np.array([get_atom_coord(r, atom) for r in residues], dtype=np.float32))
    atom_coords["CBCA"] = torch.tensor(np.array([get_cb_or_ca_coord(r) for r in residues], dtype=np.float32))

    place_missing_cb_o(atom_coords)

    return atom_coords


def get_pdb_chain_seq(
    pdb_file,
    chain_id,
):
    for chain in _parse_structure(pdb_file).get_chains():
        if chain.id == chain_id:
            return "".join([seq1(r.get_resname()) for r in chain.get_residues()])

    return None


def cdr_indices(
    chothia_pdb_file,
    cdr,
    offset_heavy=True,
):
    """Gets the index of a given CDR loop in a Chothia-numbered PDB file with chains H and L."""
    cdr = str.lower(cdr)
    if cdr not in CDR_CHOTHIA_RANGES:
        raise ValueError(f"Unknown CDR {cdr!r}; expected one of {list(CDR_CHOTHIA_RANGES)}.")

    chothia_range = CDR_CHOTHIA_RANGES[cdr]
    chain_id = cdr[0].upper()

    cdr_chain_structure = None
    for chain in _parse_structure(chothia_pdb_file).get_chains():
        if chain.id == chain_id:
            cdr_chain_structure = chain
            break
    if cdr_chain_structure is None:
        raise ValueError(f"PDB file {chothia_pdb_file} has no chain {chain_id!r}, required for CDR {cdr}.")

    residue_id_nums = [res.get_id()[1] for res in cdr_chain_structure]

    # Binary search to find the start and end of the CDR loop
    cdr_start = bisect_left(residue_id_nums, chothia_range[0])
    cdr_end = bisect_right(residue_id_nums, chothia_range[1]) - 1

    if chain_id == "L" and offset_heavy:
        heavy_seq = get_pdb_chain_seq(chothia_pdb_file, chain_id="H") or ""
        cdr_start += len(heavy_seq)
        cdr_end += len(heavy_seq)

    return cdr_start, cdr_end


def get_cdr_range_dict(
    chothia_pdb_file,
    heavy_only=False,
    light_only=False,
    offset_heavy=True,
):
    cdr_names = list(CDR_CHOTHIA_RANGES)
    if heavy_only:
        cdr_names = cdr_names[:3]
    if light_only:
        cdr_names = cdr_names[3:]

    return {cdr: cdr_indices(chothia_pdb_file, cdr, offset_heavy=offset_heavy) for cdr in cdr_names}


CIF_EXTENSIONS = (".cif", ".mmcif")
PDB_EXTENSIONS = (".pdb", ".ent")


def output_format(path: str) -> str:
    """'cif' or 'pdb', chosen from the file extension (default pdb)."""
    ext = splitext(path)[1].lower()
    if ext in CIF_EXTENSIONS:
        return "cif"
    if ext in PDB_EXTENSIONS or ext == "":
        return "pdb"
    raise ValueError(f"Unsupported structure file extension {ext!r}; use .pdb, .cif or .mmcif.")


def build_structure(
    coords: torch.Tensor,
    seq: str,
    chains: List[str],
    delim: List[int],
    bfactor: torch.Tensor = None,
    atoms=("N", "CA", "C", "CB", "O"),
    structure_id: str = "igfold",
):
    """
    Build a Bio.PDB Structure from backbone coordinates (residues x atoms x 3).

    :param seq: concatenated one-letter sequence of all chains.
    :param chains: chain ids, one per chain (single character each).
    :param delim: cumulative residue count at the end of each chain.
    :param bfactor: per-residue value written to the B-factor column (IgFold's predicted RMSD).
    """
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    delim = list(delim)
    if delim[-1] != len(seq):
        raise ValueError(f"Chain delimiters {delim} do not cover the sequence of length {len(seq)}.")
    if len(chains) < len(delim):
        raise ValueError(f"{len(delim)} chains in coordinates but only {len(chains)} chain ids given.")
    for c in chains:
        if not (isinstance(c, str) and len(c) == 1 and c.isalnum()):
            raise ValueError(f"Chain id {c!r} must be a single alphanumeric character for PDB output.")
    if coords.shape[0] != len(seq) or coords.shape[1] != len(atoms):
        raise ValueError(f"coords has shape {tuple(coords.shape)}; expected ({len(seq)}, {len(atoms)}, 3).")

    coords = coords.detach().cpu().numpy().astype(np.float32)
    bfactor = torch.zeros(len(seq)) if not exists(bfactor) else torch.as_tensor(bfactor).detach().cpu()

    structure = Structure(structure_id)
    model = Model(0)
    structure.add(model)

    serial, chain_start = 1, 0
    for chain_num, chain_end in enumerate(delim):
        chain = Chain(chains[chain_num])
        model.add(chain)
        for r in range(chain_start, chain_end):
            if seq[r] not in _aa_1_3_dict or seq[r] == "-":
                raise ValueError(f"Cannot write residue {seq[r]!r} at position {r + 1}: not a standard amino acid.")
            resname = _aa_1_3_dict[seq[r]]
            residue = Residue((" ", r - chain_start + 1, " "), resname, "    ")
            chain.add(residue)
            for a, name in enumerate(atoms):
                if resname == "GLY" and name == "CB":
                    continue
                residue.add(
                    Atom(
                        name,
                        coords[r, a],
                        round(float(bfactor[r]), 2),
                        1.0,
                        " ",
                        f" {name:<3s}",
                        serial,
                        element=name[0],
                    )
                )
                serial += 1
        chain_start = chain_end

    return structure


def write_structure(structure, path: str) -> str:
    """Write a Bio.PDB Structure as PDB or mmCIF depending on the file extension."""
    from Bio.PDB.mmcifio import MMCIFIO

    if output_format(path) == "cif":
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(path)
    else:
        io = PDBIO()
        io.set_structure(structure)
        io.save(path)
        fix_atom_serials(path)

    return path


def structure_to_pdb_string(structure) -> str:
    from io import StringIO

    io = PDBIO()
    io.set_structure(structure)
    buf = StringIO()
    io.save(buf)
    return buf.getvalue()


def convert_structure_file(in_path: str, out_path: str) -> str:
    """Convert between PDB and mmCIF (format chosen from each extension)."""
    from Bio.PDB.MMCIFParser import MMCIFParser

    if output_format(in_path) == "cif":
        structure = MMCIFParser(QUIET=True).get_structure(splitext(basename(in_path))[0], in_path)[0]
    else:
        structure = _parse_structure(in_path)

    return write_structure(structure, out_path)


def save_PDB(
    out_pdb: str,
    coords: torch.Tensor,
    seq: str,
    chains: List[str] = None,
    error: torch.Tensor = None,
    delim: Union[int, List[int]] = None,
    atoms=("N", "CA", "C", "CB", "O"),
    write_pdb=True,
) -> str:
    """Write backbone coords to a PDB (or mmCIF) file and return the PDB-format string
    (convenience wrapper around :func:`build_structure` and :func:`write_structure`)."""
    if not exists(chains):
        chains = ["H", "L"]
    if not exists(delim):
        delim = [len(seq)]
    elif isinstance(delim, int):
        delim = [delim, len(seq)]

    structure = build_structure(coords, seq, chains, delim, bfactor=error, atoms=atoms)
    if write_pdb:
        write_structure(structure, out_pdb)

    return structure_to_pdb_string(structure)


def write_pdb_bfactor(
    in_pdb_file,
    out_pdb_file,
    bfactor,
    b_chain=None,
):
    """Set the per-residue B-factor column (used by IgFold to store predicted RMSD)."""
    structure = _parse_structure(in_pdb_file)

    i = 0
    for chain in structure.get_chains():
        if exists(b_chain) and chain.id != b_chain:
            continue

        for r in chain.get_residues():
            for a in r.get_atoms():
                a.set_bfactor(float(bfactor[i]))
            i += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)
    fix_atom_serials(out_pdb_file)


def fix_atom_serials(pdb_file):
    """Renumber ATOM/HETATM/TER serial numbers consecutively and remap CONECT records."""
    with open(pdb_file) as f:
        lines = f.readlines()

    serial_map, serial, changed = {}, 1, False
    new_lines = []
    for line in lines:
        rec = line[:6]
        if rec in ("ATOM  ", "HETATM", "TER   "):
            old = line[6:11].strip()
            if rec != "TER   " and old.isdigit():
                serial_map[int(old)] = serial  # CONECT only ever references atoms
            new_line = f"{rec}{serial:5d}{line[11:]}"
            changed |= new_line != line
            line = new_line
            serial += 1
        new_lines.append(line)

    if not changed:
        return

    out = []
    for line in new_lines:
        if line.startswith("CONECT"):
            body = line.rstrip("\n")[6:]
            fields = [body[i : i + 5] for i in range(0, len(body), 5)]
            fields = [f"{serial_map.get(int(f), int(f)):5d}" if f.strip().isdigit() else f for f in fields]
            line = "CONECT" + "".join(fields) + "\n"
        out.append(line)

    with open(pdb_file, "w") as f:
        f.writelines(out)
