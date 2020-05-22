import torch
import shutil
import unittest
from lstm_cds.loader import load_genbank, divide_sequence
from .setup_dirs import setup_dirs


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.indir, self.workdir, self.outdir = setup_dirs(fpath=__file__)

    def tearDown(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def test_load_genbank(self):

        X, Y = load_genbank(gbks='./data/GCF_000008525.1_ASM852v1_genomic.gbff')

        self.assertTupleEqual((1, 3335734, 4), X.size())
        self.assertTupleEqual((1, 3335734, 1), Y.size())

    def test_divide_sequence(self):

        X = torch.randn(1, 100, 4)
        Y = torch.randn(1, 100, 1)

        X, Y = divide_sequence(X=X, Y=Y, seq_len=3)

        self.assertTupleEqual((34, 3, 4), X.size())
        self.assertTupleEqual((34, 3, 1), Y.size())
