import numpy as np
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


def get_cds_labels(
        chromosome: ngs.Chromosome,
        label_length: Optional[int]) -> torch.Tensor:
    """
    For each genome position, label whether it's CDS or not (0 or 1)

    Returns:
        1D tensor, size=(len(chromosome), ), dtype=torch.long
    """
    labels = torch.zeros(len(chromosome.sequence), dtype=torch.long)

    for feature in chromosome.feature_array:
        if feature.type != 'CDS':
            continue
        if feature.strand == '-':
            continue

        from_ = feature.start - 1
        to_ = (from_ + label_length) if label_length else feature.end

        labels[from_: to_] = 1

    return labels.long()


def load_genbank(
        gbks: Union[str, List[str]],
        label_length: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For X, DNA sequence is tokenized to 4 nucleotides [A, C, G, T]

    For Y, CDS on '+' strand is labeled as 1, otherwise 0

    Forward and reverse strands were concatenated, thus seq_len = all DNA len * 2

    Args:
        gbks:
            Path(s) to genbank file(s)

        label_length:
            Length to be labeled positive from the start nucleotide of a CDS
            If None, label_length = CDS length

    Returns:
        X: 2D tensor, size=(seq_len, 4), dtype=torch.float

        Y: 1D tensor, size=(seq_len, ), dtype=torch.long
    """

    if type(gbks) is str:
        gbks = [gbks]

    chromosomes = []
    for gbk in gbks:
        chromosomes += ngs.read_genbank(file=gbk)

    Xs, Ys = [], []
    for chromosome in chromosomes:

        X_f = dna_to_one_hot(seq=chromosome.sequence)
        Y_f = get_cds_labels(
            chromosome=chromosome,
            label_length=label_length)

        chromosome.reverse()

        X_r = dna_to_one_hot(seq=chromosome.sequence)
        Y_r = get_cds_labels(
            chromosome=chromosome,
            label_length=label_length)

        Xs += [X_f, X_r]
        Ys += [Y_f, Y_r]

    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)

    return X, Y


def divide_sequence(
        x: torch.Tensor,
        seq_len: int,
        pad: bool) -> torch.Tensor:
    """
    Break full_len -> n_samples * seq_len

    Args:
        x:
            tensor: (full_len, ...)

        seq_len:
            Divided sequence length, the second dimension of the output tensor

        pad:
            Pad with zeros or discard the remainder sequence

    Returns:
        tensor, where the first input dimension (full_len, ) is split into (n_samples, seq_len)
    """

    full_len = x.size()[0]
    k_dims = list(x.size()[1:])

    remainder = full_len % seq_len
    divisible = remainder == 0

    if not divisible:
        if pad:
            pad_len = seq_len - remainder

            pad_size = [pad_len] + k_dims
            x_pad = torch.zeros(size=pad_size, dtype=x.dtype)

            if x.is_cuda:
                x_pad = x_pad.cuda()

            x = torch.cat([x, x_pad], dim=0)

        else:  # discard remainder
            x = x[0:-remainder]

    new_size = [-1, seq_len] + k_dims

    return x.view(*new_size)


def split(
        x: torch.Tensor,
        training_fraction: float,
        dim: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split into training and test sets on the <dim> dimension
    """

    n_samples = x.size()[0]

    train_size = int(n_samples * training_fraction)
    test_size = n_samples - train_size

    x_train, x_test = torch.split(x, [train_size, test_size], dim=dim)

    return x_train, x_test


def shuffle(
        X: torch.Tensor,
        Y: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:

    n_samples = X.shape[0]
    rand_order = torch.randperm(n_samples)

    X = X[rand_order]
    Y = Y[rand_order]

    return X, Y
