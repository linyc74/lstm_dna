import os
from ngslite import orf_finder, get_files, genbank_to_fasta, make_genbank, call


def main():

    srcdir = '../data'

    gbks = get_files(
        source=srcdir,
        endswith='gbff',
        isfullpath=False)

    for gbk in gbks:
        fna = gbk[:-len('.gbff')] + '.fna'
        genbank_to_fasta(
            file=os.path.join(srcdir, gbk),
            output=fna)

    fnas = get_files(
        source='.',
        endswith='.fna',
        isfullpath=False)

    for fna in fnas:
        gtf = fna[:-len('.fna')] + '.gtf'
        orf_finder(
            fasta=fna,
            output=gtf,
            min_length=50)

    gtfs = get_files(
        source='.',
        endswith='.gtf',
        isfullpath=False)

    for fna, gtf, gbk in zip(fnas, gtfs, gbks):
        make_genbank(
            fasta=fna,
            gtf=gtf,
            output=f'{__file__[:-3]}_{gbk}',
            shape='circular')

    call('rm *.fna *.gtf')
    os.makedirs('outdir', exist_ok=True)
    call('mv *.gbff outdir/')


if __name__ == '__main__':
    main()
