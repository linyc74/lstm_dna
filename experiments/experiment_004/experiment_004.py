import os
import torch
from ngslite import get_files, read_genbank, write_genbank
from lstm_dna.model import LSTMModel
from lstm_dna.predictor import Predictor
from lstm_dna.annotate import CDSAnnotator


EXPERIMENT_NAME = f'{os.path.basename(__file__)[:-3]}'


def load_model(file: str, cuda: bool) -> LSTMModel:

    device = 'cuda' if cuda else 'cpu'

    model = torch.load(file, map_location=torch.device(device))
    model.is_cuda = cuda

    return model


def main():

    model_file = '../experiment_003/models/experiment_003_model_3.model'
    gbkdir = '../data'
    cuda = True
    min_protein_len = 50
    outdir = './outdir'

    model = load_model(file=model_file, cuda=cuda)
    predictor = Predictor(model=model)

    gbks = get_files(source=gbkdir, isfullpath=True)

    for gbk in gbks:

        chromosomes = read_genbank(file=gbk)
        new_chromosomes = []

        for chromosome in chromosomes:

            annotator = CDSAnnotator(
                predictor=predictor,
                min_protein_len=min_protein_len)

            c = annotator.annotate(
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
