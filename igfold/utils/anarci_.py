import warnings
import re
from anarci import number
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1


# Jumping through some hoops to catch ANARCI edge cases.
def get_anarci_numbering(seq, scheme="chothia"):
    numbering, _ = number(seq, scheme=scheme)

    anarci_seq = "".join(s[1] for s in numbering)
    gap_pos = [(c.start() + 1, c.end() - 1)
               for c in re.finditer("[A-Z]-+[A-Z]", anarci_seq)]
    if len(gap_pos) > 0:
        for gap_start, gap_end in reversed(gap_pos):
            del numbering[gap_start:gap_end]

    numbering_pad = 10
    if len(numbering) < len(seq):
        for i in range(numbering_pad):
            numbering.insert(0, ((numbering[0][0][0] - 1, ' '), "-"))
            numbering.append(((numbering[-1][0][0] + 1, ' '), "-"))

    if len(seq) != len(numbering):
        anarci_seq = "".join(s[1] for s in numbering)
        reiter = re.finditer(anarci_seq.strip("-"), seq)
        pdb_ali = [(c.start(), c.end()) for c in reiter][0]
        reiter = re.finditer(seq[pdb_ali[0]:pdb_ali[1]], anarci_seq)
        anarci_ali = [(c.start(), c.end()) for c in reiter][0]

        for pdb_i, anarci_i in zip(reversed(range(pdb_ali[0])),
                                   reversed(range(anarci_ali[0]))):
            numbering[anarci_i] = (numbering[anarci_i][0], seq[pdb_i])
        for pdb_i, anarci_i in zip(range(pdb_ali[1], len(seq)),
                                   range(anarci_ali[1], len(anarci_seq))):
            numbering[anarci_i] = (numbering[anarci_i][0], seq[pdb_i])

    return numbering


def renumber_pdb(
    in_pdb_file,
    out_pdb_file=None,
    scheme="chothia",
):
    """
    Renumber the pdb file.
    """
    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(
            "_",
            in_pdb_file,
        )

    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain]))
        numbering = get_anarci_numbering(seq, scheme=scheme)
        anarci_res_i = 0
        for pdb_i, pdb_r in enumerate(chain.get_residues()):
            if anarci_res_i >= len(numbering):
                chain.__delitem__(pdb_r.get_id())
                continue

            while numbering[anarci_res_i][1] == "-":
                anarci_res_i += 1

            res_num, res_aa = numbering[anarci_res_i]
            if res_aa != seq1(pdb_r.get_resname()):
                raise Exception(f"Failed to renumber PDB file {in_pdb_file}")
            pdb_r._id = (' ', *res_num)
            anarci_res_i += 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)
