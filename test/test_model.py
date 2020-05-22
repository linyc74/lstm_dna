import shutil
import unittest
from lstm_cds.model import LSTMModel
from .setup_dirs import setup_dirs


class TestLSTMModel(unittest.TestCase):

    def setUp(self):
        self.indir, self.workdir, self.outdir = setup_dirs(fpath=__file__)

    def tearDown(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def test___init__(self):
        lstm_model = LSTMModel(
            sizes=[1000, 128, 64, 32, 10],
            batch_first=True,
            bidirectional=True)

        expected = f'''\
LSTMModel(
  (lstm_1): LSTM(1000, 128, batch_first=True, bidirectional=True)
  (lstm_2): LSTM(256, 64, batch_first=True, bidirectional=True)
  (lstm_3): LSTM(128, 32, batch_first=True, bidirectional=True)
  (fc): Linear(in_features=64, out_features=10, bias=True)
)'''

        self.assertEqual(expected, str(lstm_model))
