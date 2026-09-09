from Bio import SeqIO


def get_fasta_chain_seq(
    fasta_file,
    chain_id,
):
    for chain in SeqIO.parse(fasta_file, "fasta"):
        if chain.id == chain_id or chain.id.endswith(f":{chain_id}"):
            return str(chain.seq)


def get_fasta_chain_dict(fasta_file):
    seq_dict = {}
    for chain in SeqIO.parse(fasta_file, "fasta"):
        # accept both ">H" and ">1abc:H" style record ids
        chain_id = chain.id.rsplit(":", 1)[-1]
        seq_dict[chain_id] = str(chain.seq)

    return seq_dict
