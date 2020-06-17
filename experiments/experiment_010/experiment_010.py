import numpy as np
import pandas as pd
from typing import Tuple
from ngslite import read_genbank, get_files, Chromosome


def get_codon_arr(chromosome: Chromosome) -> np.ndarray:
    """
    On the forward strand, mark the first base of each codon as 1
    Because only the first base is labeled 1, overlapping codons of the 3 frames can be distinguished
    """

    seq_len = len(chromosome.sequence)
    arr = np.zeros((seq_len - 2,), dtype=np.int)

    for f in chromosome.features:

        if f.type != 'CDS':
            continue
        if f.strand == '-':
            continue

        protein_len = (f.end - f.start) // 3
        for aa in range(protein_len):
            pos = f.start + (aa * 3) - 1  # -1 to 0-based
            arr[pos] = 1

    return arr


def compare_chromosomes(
        left_chr: Chromosome,
        right_chr: Chromosome) -> Tuple[int, int, int]:
    """
    Returns:
        left: count of codons only appearing in left_chr

        right: count of codons only appearing in right_chr

        inner: count of codons in both left_chr and right_chr
    """

    assert len(left_chr.sequence) == len(right_chr.sequence)

    left, right, inner = 0, 0, 0

    arr_l = get_codon_arr(chromosome=left_chr)
    arr_r = get_codon_arr(chromosome=right_chr)

    left += np.sum((arr_l - arr_r) == 1)
    right += np.sum((arr_r - arr_l) == 1)
    inner += np.sum((arr_l + arr_r) == 2)

    left_chr.reverse()
    right_chr.reverse()

    arr_l = get_codon_arr(chromosome=left_chr)
    arr_r = get_codon_arr(chromosome=right_chr)

    left += np.sum((arr_l - arr_r) == 1)
    right += np.sum((arr_r - arr_l) == 1)
    inner += np.sum((arr_l + arr_r) == 2)

    return left, right, inner


def get_precision_recall(
        true_gbk: str,
        predicted_gbk: str) -> Tuple[float, float]:

    left, inner, right = 0, 0, 0

    true_chromosomes = read_genbank(true_gbk)
    pred_chromosomes = read_genbank(predicted_gbk)

    for true_chr, pred_chr in zip(true_chromosomes, pred_chromosomes):
        l, r, i = compare_chromosomes(left_chr=true_chr, right_chr=pred_chr)
        left += l
        right += r
        inner += i

    precision = inner / (inner + right)
    recall = inner / (inner + left)

    return precision, recall


def main():

    true_gbk_dir = '../data'
    lstm_gbk_dir = '../experiment_006/outdir'
    orffinder_gbk_dir = '../experiment_009/outdir'
    output_csv = f'{__file__[:-3]}.csv'

    columns = [
        'Species',
        'LSTM Precision',
        'LSTM Recall',
        'ORFfinder Precision',
        'ORFfinder Recall'
    ]

    seqname_to_species = {
        'NC_000913': 'Escherichia coli',
        'NC_002505': 'Vibrio cholerae',
        'NC_002516': 'Pseudomonas aeruginosa',
        'NC_003098': 'Streptococcus pneumoniae',
        'NC_004668': 'Enterococcus faecalis',
        'NC_000915': 'Helicobacter pylori',
        'NC_000964': 'Bacillus subtilis',
        'NC_009089': 'Clostridioides difficile',
        'NC_010729': 'Porphyromonas gingivalis',
        'NC_007795': 'Staphylococcus aureus',
        'NC_000962': 'Mycobacterium tuberculosis',
        'NC_003198': 'Salmonella enterica',
        'NC_003888': 'Streptomyces coelicolor',
        'NC_016845': 'Klebsiella pneumoniae',
        'NZ_CP009257': 'Acinetobacter baumannii',
    }

    fnames = get_files(
        source=true_gbk_dir,
        endswith='gbff',
        isfullpath=False)

    df = pd.DataFrame(columns=columns)

    for fname in fnames:

        true_gbk = get_files(
            source=true_gbk_dir, endswith=fname, isfullpath=True)[0]

        lstm_gbk = get_files(
            source=lstm_gbk_dir, endswith=fname, isfullpath=True)[0]

        orffinder_gbk = get_files(
            source=orffinder_gbk_dir, endswith=fname, isfullpath=True)[0]

        seqname = read_genbank(true_gbk)[0].seqname
        species = seqname_to_species[seqname]

        lstm_precision, lstm_recall = get_precision_recall(
            true_gbk=true_gbk,
            predicted_gbk=lstm_gbk)

        orffinder_precision, orffinder_recall = get_precision_recall(
            true_gbk=true_gbk,
            predicted_gbk=orffinder_gbk)

        data = [
            species,
            lstm_precision,
            lstm_recall,
            orffinder_precision,
            orffinder_recall
        ]

        row = {key: val for key, val in zip(columns, data)}

        df = df.append(row, ignore_index=True)

    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
