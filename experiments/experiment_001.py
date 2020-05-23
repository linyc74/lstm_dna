import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lstm_dna.model import LSTMModel
from lstm_dna.loader import load_genbank, divide_sequence
from lstm_dna.functional import binary_accuracy


class Trainer:

    epoch: int

    def __init__(
            self,
            model,
            X: torch.Tensor,
            Y: torch.Tensor,
            criterion,
            optimizer,
            writer: SummaryWriter):

        self.model = model
        self.X = X
        self.Y = Y
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer

    def log(self, tag: str, loss: float, accuracy: float):

        self.writer.add_scalar(
            tag=f'Loss ({tag})', scalar_value=loss, global_step=self.epoch)

        self.writer.add_scalar(
            tag=f'Accuracy ({tag})', scalar_value=accuracy, global_step=self.epoch)

    def train(self, epochs: int, batch_size: int):

        n_samples, seq_len, input_size = self.X.size()

        for epoch in range(epochs):

            self.epoch = epoch

            i = 0
            losses = []
            accuracies = []
            while i < n_samples:
                x = self.X[i:(i + batch_size), :, :]
                y = self.Y[i:(i + batch_size), :, :]

                self.optimizer.zero_grad()

                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()

                y_hat = torch.sigmoid(y_hat)
                accuracy = binary_accuracy(y_hat, y)

                self.log(
                    tag='mini-batches',
                    loss=loss.item(),
                    accuracy=accuracy)

                losses.append(loss.item())
                accuracies.append(accuracy)

                i += batch_size

            self.log(
                tag='epochs',
                loss=sum(losses) / len(losses),
                accuracy=sum(accuracies) / len(accuracies)
            )

        return self.model


def main():

    experiment_name = f'{os.path.basename(__file__)[:-3]}'
    gbk = '../data/GCF_000746645.1_ASM74664v1_genomic.gbff'
    seq_len = 100
    cuda = True
    learning_rate = 1e-3
    epochs = 100
    batch_size = 1000

    # Data
    X, Y = load_genbank(gbks=gbk, cuda=cuda)
    X, Y = divide_sequence(X=X, Y=Y, seq_len=seq_len)

    model = LSTMModel(
        sizes=[4, 128, 64, 1],
        batch_first=True,
        bidirectional=True,
        cuda=cuda)

    # Model
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Tensorboard
    writer = SummaryWriter(log_dir=f'runs/{experiment_name}')

    # Training
    trainer = Trainer(
        model=model,
        X=X,
        Y=Y,
        criterion=criterion,
        optimizer=optimizer,
        writer=writer)

    model = trainer.train(epochs=epochs, batch_size=batch_size)

    # Finish
    torch.save(model, f'./models/{experiment_name}.model')
    writer.close()


if __name__ == '__main__':
    main()
