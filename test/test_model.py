import torch
import shutil
import unittest
from lstm_dna.model import LSTMModel
from .setup_dirs import setup_dirs


class TestLSTMModel(unittest.TestCase):

    def setUp(self):
        self.indir, self.workdir, self.outdir = setup_dirs(fpath=__file__)

        self.lstm_model = LSTMModel(
            sizes=[1000, 128, 64, 32, 10],
            batch_first=True,
            bidirectional=True,
            output_activation='softmax',
            cuda=False)

    def tearDown(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def test___init__(self):
        expected = f'''\
LSTMModel(
  (lstm_1): LSTM(1000, 128, batch_first=True, bidirectional=True)
  (lstm_2): LSTM(256, 64, batch_first=True, bidirectional=True)
  (lstm_3): LSTM(128, 32, batch_first=True, bidirectional=True)
  (fc): Linear(in_features=64, out_features=10, bias=True)
)'''
        self.assertEqual(expected, str(self.lstm_model))

    def test_forward(self):
        x = torch.randn(10, 100, 1000)
        y_hat = self.lstm_model(x)
        self.assertTupleEqual((10, 100, 10), y_hat.size())
