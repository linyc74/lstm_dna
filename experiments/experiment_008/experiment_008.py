from ngslite import read_genbank, write_genbank


def main():

    gbk1 = '../data/GCF_000005845.2_ASM584v2_genomic.gbff'
    gbk2 = '../experiment_006/outdir/experiment_006_GCF_000005845.2_ASM584v2_genomic.gbff'

    chromosomes1 = read_genbank(gbk1)
    chromosomes2 = read_genbank(gbk2)

    for chr1, chr2 in zip(chromosomes1, chromosomes2):
        for f in chr2.features:
            f.add_attribute(key='note', val='"color: #00ff00"')
            chr1.features.append(f)

    write_genbank(
        data=chromosomes1,
        file=f'{chromosomes1[0].seqname}.gbk')


if __name__ == '__main__':
    main()
