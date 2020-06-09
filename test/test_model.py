import torch
import shutil
import unittest
from lstm_dna.model import LSTMModel
from .setup_dirs import setup_dirs


class TestLSTMModel(unittest.TestCase):

    def setUp(self):
        self.indir, self.workdir, self.outdir = setup_dirs(fpath=__file__)

        self.model = LSTMModel(
            sizes=[4, 128, 64, 32, 2],
            bidirectional=True,
            softmax_output=True,
            cuda=False)

    def tearDown(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def test___init__(self):
        expected = f'''\
LSTMModel(
  (lstm_layers): ModuleList(
    (0): LSTM(4, 128, batch_first=True, bidirectional=True)
    (1): LSTM(256, 64, batch_first=True, bidirectional=True)
    (2): LSTM(128, 32, batch_first=True, bidirectional=True)
  )
  (fc): Linear(in_features=64, out_features=2, bias=True)
)'''
        self.assertEqual(expected, str(self.model))

    def test_forward(self):
        x = torch.randn(10, 100, 4)  # (batch_size, seq_len, input_size)

        y_hat = self.model(x)

        size = (10, 2, 100)  # (batch_size, output_size, seq_len)
        self.assertTupleEqual(size, y_hat.size())

