import warnings
from abnumber import Chain
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1

from igfold.utils.pdb import clean_pdb


def is_heavy(seq):
    chain = Chain(seq, scheme='chothia')

    return chain.is_heavy_chain()


def rechain_pdb(pdb_file):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("_", pdb_file)

    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain]))
        abnum_chain = Chain(seq, scheme='chothia')
        chain_id = "H" if abnum_chain.is_heavy_chain() else "L"
        try:
            chain.id = chain_id
        except ValueError:
            chain.id = chain_id + "_"
    for chain in structure.get_chains():
        if "_" in chain.id:
            chain.id = chain.id.replace("_", "")

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)


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

    clean_pdb(in_pdb_file)

    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(
            "_",
            in_pdb_file,
        )

    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain]))
        abnum_chain = Chain(seq, scheme=scheme)
        numbering = abnum_chain.positions.items()

        chain_res = list(chain.get_residues())
        assert len(chain_res) == len(numbering)

        for pdb_r, (pos, aa) in zip(chain_res, numbering):
            if aa != seq1(pdb_r.get_resname()):
                raise Exception(f"Failed to renumber PDB file {in_pdb_file}")
            pos = str(pos)[1:]
            if not pos[-1].isnumeric():
                ins = pos[-1]
                pos = int(pos[:-1])
            else:
                pos = int(pos)
                ins = ' '

            pdb_r._id = (' ', pos, ins)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)


def truncate_seq(seq, scheme="chothia"):
    abnum_chain = Chain(seq, scheme=scheme)
    numbering = abnum_chain.positions.items()
    seq = "".join([r[1] for r in list(numbering)])

    return seq