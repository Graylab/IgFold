from Bio import SeqIO, pairwise2


def get_fasta_chain_seq(
    fasta_file,
    chain_id,
):
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        if ":{}".format(chain_id) in chain.id:
            return str(chain.seq)


def get_fasta_chain_dict(fasta_file):
    seq_dict = {}
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        seq_dict[chain.id] = str(chain.seq)

    return seq_dict


def pairwise_align(
    seq1,
    seq2,
):
    ###
    # Aligns two sequences using the Needleman-Wunsch algorithm
    # Returns alignment of seq2 into seq1
    ###
    ali = pairwise2.align.globalxx(
        seq1,
        seq2,
    )[0]
    ali_list = []
    seq1_i, seq2_i = 0, 0
    for ali_seq in ali.seqB.split("-"):
        if len(ali_seq) == 0:
            seq1_i += 1
        else:
            l = len(ali_seq)
            ali_list.append((seq1_i, seq1_i + l, seq2_i, seq2_i + l))
            seq1_i += l

    return ali_list