import torch
import shutil
import unittest
from lstm_dna.loader import load_genbank, divide_sequence
from .setup_dirs import setup_dirs


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.indir, self.workdir, self.outdir = setup_dirs(fpath=__file__)

    def tearDown(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def test_load_genbank(self):

        X, Y = load_genbank(
            gbks=f'{self.indir}/GCF_000008525.1_ASM852v1_genomic.gbff',
            label_length=90)

        self.assertTupleEqual((3335734, 4), X.size())
        self.assertTupleEqual((3335734, ), Y.size())
        self.assertEqual(torch.float, X.dtype)
        self.assertEqual(torch.long, Y.dtype)

    def test_divide_sequence(self):

        x = torch.randn(100, 4)

        for pad, expected in [
            (True, (34, 3, 4)),
            (False, (33, 3, 4))
        ]:
            size = divide_sequence(x, seq_len=3, pad=pad).size()
            self.assertTupleEqual(expected, size)
