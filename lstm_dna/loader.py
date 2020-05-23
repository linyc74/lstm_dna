import torch
import torch.nn.functional as F
import ngslite as ngs
from typing import Tuple, Optional, Union, List


def dna_to_one_hot(seq: str) -> torch.Tensor:
    """
    Convert DNA sequence into one-hot encoding tensor

    Returns:
        2D tensor, size=(len(seq), 4), dtype=float
    """
    base_to_idx = {'a': 0, 'c': 1, 'g': 2, 't': 3}

    x = []
    for b in seq.lower():
        x.append(base_to_idx.get(b, 0))

    return F.one_hot(torch.tensor(x)).float()


def get_cds_labels(chromosome: ngs.Chromosome) -> torch.Tensor:
    """
    For each genome position, label whether it's CDS or not (0 or 1)

    Returns:
        1D tensor, size=(len(chromosome), ), dtype=float
    """
    labels = torch.zeros(len(chromosome.sequence))

    for feature in chromosome.feature_array:
        if feature.type != 'CDS':
            continue
        if feature.strand == '-':
            continue
        start = feature.start - 1
        end = feature.end
        labels[start: end] = 1

    return labels


def load_genbank(
        gbks: Union[str, List[str]],
        cuda: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert genbank file(s) to LSTM 3D tensors

    For X, DNA sequence is tokenized to 4 nucleotides [A, C, G, T]

    For Y, CDS on '+' strand is labeled as 1, otherwise 0

    Forward and reverse strands were concatenated, thus seq_len = all DNA len * 2

    Args:
        gbks:
            Path(s) to genbank file(s)

        cuda:
            GPU or not

    Returns:
        X: 3D tensor, size=(1, seq_len, 4), dtype=float

        Y: 3D tensor, size=(1, seq_len, 1), dtype=float
    """

    if type(gbks) is str:
        gbks = [gbks]

    chromosomes = []
    for gbk in gbks:
        chromosomes += ngs.read_genbank(file=gbk)

    Xs, Ys = [], []
    for chromosome in chromosomes:

        X_f = dna_to_one_hot(seq=chromosome.sequence)
        Y_f = get_cds_labels(chromosome=chromosome)

        chromosome.reverse()

        X_r = dna_to_one_hot(seq=chromosome.sequence)
        Y_r = get_cds_labels(chromosome=chromosome)

        Xs += [X_f, X_r]
        Ys += [Y_f, Y_r]

    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)

    X = X.view(1, -1, 4)  # 2D -> 3D
    Y = Y.view(1, -1, 1)  # 1D -> 3D

    if cuda:
        X = X.cuda()
        Y = Y.cuda()

    return X, Y


def divide_sequence(
        X: torch.tensor,
        Y: torch.tensor,
        seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Break full_len -> n_samples * seq_len

    Args:
        X: 3D tensor, size (1, full_len, 4), dtype=float

        Y: 3D tensor, size (1, full_len, 1), dtype=float

        seq_len: input sequence length for the LSTM model

    Returns:
        X: 3D tensor, size (n_samples, seq_len, 4), dtype=float

        Y: 3D tensor, size (n_samples, seq_len, 1), dtype=float
    """

    _, full_len, input_size = X.size()
    _, full_len, output_size = Y.size()

    quotient = full_len // seq_len
    remainder = full_len % seq_len

    if remainder == 0:
        n_samples = quotient
    else:
        n_samples = quotient + 1
        pad_len = seq_len - remainder

        X_pad = torch.zeros(1, pad_len, input_size, dtype=torch.float)
        if X.is_cuda:
            X_pad = X_pad.cuda()

        Y_pad = torch.zeros(1, pad_len, output_size, dtype=torch.float)
        if Y.is_cuda:
            Y_pad = Y_pad.cuda()

        X = torch.cat([X, X_pad], dim=1)
        Y = torch.cat([Y, Y_pad], dim=1)

    X = X.view(n_samples, seq_len, 4)
    Y = Y.view(n_samples, seq_len, 1)

    return X, Y
