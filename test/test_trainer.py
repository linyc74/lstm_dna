import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import unittest
from lstm_dna.model import LSTMModel
from lstm_dna.trainer import Trainer
from lstm_dna.functional import softmax_accuracy
from .setup_dirs import setup_dirs


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.indir, self.workdir, self.outdir = setup_dirs(fpath=__file__)

        torch.manual_seed(0)

        self.model = LSTMModel(
            sizes=[4, 128, 2],
            bidirectional=True,
            softmax_output=True,
            cuda=False)

        X_train = (torch.rand(100, 32, 4) > 0.5).float()  # (n_samples, seq_len, input_size)
        Y_train = (torch.rand(100, 32) > 0.5).long()  # (n_samples, seq_len)
        X_test = (torch.rand(50, 32, 4) > 0.5).float()
        Y_test = (torch.rand(50, 32) > 0.5).long()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        self.trainer = Trainer(
            model=self.model,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            criterion=criterion,
            optimizer=optimizer,
            accuracy=softmax_accuracy,
            mini_batch_size=16,
            log_dir=f'{self.workdir}/tensorboard/test')

    def tearDown(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def test_train(self):
        self.trainer.train(
            max_epochs=2,
            overtrain_epochs=2)

    def test_validate(self):
        self.trainer.validate()
