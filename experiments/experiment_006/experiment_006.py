import os
import torch
from ngslite import get_files, read_genbank, write_genbank, rev_comp, Chromosome
from lstm_dna.model import LSTMModel
from lstm_dna.annotate import CDSAnnotator
from lstm_dna.predictor import Predictor, binary_output_to_label


EXPERIMENT_NAME = f'{os.path.basename(__file__)[:-3]}'


def load_model(file: str, cuda: bool) -> LSTMModel:

    model = torch.load(file, map_location=torch.device('cpu'))

    # Because source code was changed, need to unpack the parameters and
    #   load them into a newly instantiated LSTMModel
    state_dict = {}
    for key, val in model.state_dict().items():
        k = key.replace('_1', '_layers.0').replace('_2', '_layers.1')
        state_dict[k] = val

    new_model = LSTMModel(
        sizes=[4, 128, 64, 1],
        bidirectional=True,
        softmax_output=False,
        cuda=cuda)

    new_model.load_state_dict(state_dict)

    return new_model


def start_codons():

    gbkdir = '../data'
    gbks = get_files(source=gbkdir, isfullpath=True)
    start_codon_dict = {}

    for gbk in gbks:
        chromosomes = read_genbank(file=gbk)

        for chromosome in chromosomes:
            seq = chromosome.sequence

            for f in chromosome.features:
                if f.type != 'CDS':
                    continue
                if f.strand == '+':
                    start_codon = seq[f.start - 1:f.start + 2]
                else:
                    start_codon = rev_comp(seq[f.end-3:f.end])

                start_codon_dict.setdefault(start_codon, 0)
                start_codon_dict[start_codon] += 1

    for codon, count in start_codon_dict.items():
        print(f'{codon}: {count}')


def main():

    model_file = '../experiment_003/models/experiment_003_model_3.model'
    gbkdir = '../data'
    cuda = torch.cuda.is_available()
    min_protein_len = 50
    outdir = './outdir'

    model = load_model(file=model_file, cuda=cuda)

    predictor = Predictor(
        model=model,
        output_to_label=binary_output_to_label)

    gbks = get_files(source=gbkdir, isfullpath=True)

    for gbk in gbks:

        chromosomes = read_genbank(file=gbk)
        new_chromosomes = []

        for chromosome in chromosomes:

            annotator = CDSAnnotator(
                predictor=predictor,
                min_protein_len=min_protein_len)

            c: Chromosome = annotator.annotate(
                dna=chromosome.sequence,
                seqname=chromosome.seqname,
                circular=chromosome.circular)

            c.genbank_locus_text = chromosome.genbank_locus_text

            new_chromosomes.append(c)

        os.makedirs(outdir, exist_ok=True)
        write_genbank(
            data=new_chromosomes,
            file=f'{outdir}/{EXPERIMENT_NAME}_{os.path.basename(gbk)}',
            use_locus_text=True)


if __name__ == '__main__':
    main()
