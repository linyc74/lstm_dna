import shutil
import unittest
from ngslite import read_genbank
from lstm_dna.annotate import DNARegion
from .setup_dirs import setup_dirs


class TestDNARegion(unittest.TestCase):

    def setUp(self):
        self.indir, self.workdir, self.outdir = setup_dirs(fpath=__file__)

    def tearDown(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def test_get_orf_regions(self):

        seq = read_genbank(f'{self.indir}/GCF_000005845.2_ASM584v2_genomic (91,413 .. 97,084).gbk')[0].sequence

        dna_region = DNARegion(
            chromosome_position=1,
            seq=seq,
            min_protein_len=50)

        orf_regions = dna_region.get_orf_regions()

        expected = [(1, 1767), (1754, 3241), (3238, 4596), (4590, 5672)]

        self.assertListEqual(expected, orf_regions)
