import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from ngslite import get_files
from lstm_dna.model import LSTMModel
from lstm_dna.loader import load_genbank, divide_sequence
from lstm_dna.trainer import Trainer


EXPERIMENT_NAME = f'{os.path.basename(__file__)[:-3]}'
TRAINING_FRACTION = 0.7
MAX_EPOCHS = 1000
OVERTRAIN_EPOCHS = 5
CUDA = True


def shuffle(
        X: torch.Tensor,
        Y: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:

    n_samples, _, _ = X.size()
    rand_order = torch.randperm(n_samples)

    X = X[rand_order, :, :]
    Y = Y[rand_order, :, :]

    return X, Y


def split(
        t: torch.Tensor,
        training_fraction: float) \
        -> Tuple[torch.Tensor, torch.Tensor]:

    n_samples, _, _ = t.size()

    train_size = int(n_samples * training_fraction)
    test_size = n_samples - train_size

    x_train, x_test = torch.split(t, [train_size, test_size], dim=0)

    return x_train, x_test


def train_model(
        stage: int,
        X: torch.Tensor,
        Y: torch.Tensor,
        seq_len: int,
        mini_batch_size: int,
        learning_rate: float,
        model: LSTMModel) -> LSTMModel:

    X, Y = divide_sequence(X=X, Y=Y, seq_len=seq_len)
    X, Y = shuffle(X=X, Y=Y)
    X_train, X_test = split(X, training_fraction=TRAINING_FRACTION)
    Y_train, Y_test = split(Y, training_fraction=TRAINING_FRACTION)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(
        log_dir=f'tensorboard/{EXPERIMENT_NAME}/stage_{stage}_seq_len_{seq_len}_lr_{learning_rate}')

    trainer = Trainer(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        criterion=criterion,
        optimizer=optimizer,
        mini_batch_size=mini_batch_size,
        writer=writer)

    model = trainer.train(
        max_epochs=MAX_EPOCHS,
        overtrain_epochs=OVERTRAIN_EPOCHS)

    writer.close()

    return model


def main():
    gbkdir = '../data'

    gbks = get_files(source=gbkdir, isfullpath=True)
    X, Y = load_genbank(gbks=gbks, cuda=False)

    model = LSTMModel(
        sizes=[4, 128, 1],
        batch_first=True,
        bidirectional=True,
        cuda=CUDA)

    for stage, seq_len, mini_batch_size, learning_rate in [
        (1, 128,  256, 1e-3),
        (2, 1024, 64,  1e-4),
        (3, 8192, 16,  1e-5),
    ]:

        model = train_model(
            stage=stage,
            X=X,
            Y=Y,
            seq_len=seq_len,
            mini_batch_size=mini_batch_size,
            learning_rate=learning_rate,
            model=model)

        torch.save(model, f'./models/{EXPERIMENT_NAME}_stage_{stage}.model')

    torch.save(model, f'./models/{EXPERIMENT_NAME}.model')


if __name__ == '__main__':
    main()
