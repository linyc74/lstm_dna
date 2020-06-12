import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib_venn import venn2
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
        chr1: Chromosome,
        chr2: Chromosome) -> Tuple[int, int, int]:
    """
    Returns:
        left: count of codons only appearing in chr1

        right: count of codons only appearing in chr2

        inner: count of codons in both chr1 and chr2
    """

    assert len(chr1.sequence) == len(chr2.sequence)

    left, right, inner = 0, 0, 0

    arr1 = get_codon_arr(chromosome=chr1)
    arr2 = get_codon_arr(chromosome=chr2)

    left += np.sum((arr1 - arr2) == 1)
    right += np.sum((arr2 - arr1) == 1)
    inner += np.sum((arr1 + arr2) == 2)

    chr1.reverse()
    chr2.reverse()

    arr1 = get_codon_arr(chromosome=chr1)
    arr2 = get_codon_arr(chromosome=chr2)

    left += np.sum((arr1 - arr2) == 1)
    right += np.sum((arr2 - arr1) == 1)
    inner += np.sum((arr1 + arr2) == 2)

    return left, right, inner


def compare_gbks(
        gbk1: str,
        gbk2: str) -> Tuple[int, int, int]:

    left, inner, right = 0, 0, 0

    chromosomes1 = read_genbank(gbk1)
    chromosomes2 = read_genbank(gbk2)

    for chr1, chr2 in zip(chromosomes1, chromosomes2):
        l, r, i = compare_chromosomes(chr1=chr1, chr2=chr2)
        left += l
        right += r
        inner += i

    return left, right, inner


def plot_venn(
        title: str,
        left: int,
        right: int,
        inner: int,
        png: str):

    fig = plt.figure(figsize=(3, 3), dpi=600)

    n = right + left + inner
    venn2(
        subsets=(left, right, inner),
        set_labels=('', ''),
        subset_label_formatter=lambda x: f"{x}\n({(x / n):.2%})"
    )

    plt.title(title)
    fig.savefig(png, fmt='png')
    plt.close()


def main():

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

    gbk1s = get_files(
        source='../data',
        endswith='gbff',
        isfullpath=True)

    gbk2s = get_files(
        source='../experiment_006/outdir',
        endswith='gbff',
        isfullpath=True)

    os.makedirs('outdir', exist_ok=True)
    for gbk1, gbk2 in zip(gbk1s, gbk2s):

        left, right, inner = compare_gbks(gbk1=gbk1, gbk2=gbk2)

        seqname = read_genbank(gbk1)[0].seqname
        title = seqname_to_species[seqname]

        plot_venn(
            title=title,
            left=left,
            right=right,
            inner=inner,
            png=f'outdir/{title}.png')


if __name__ == '__main__':
    main()
