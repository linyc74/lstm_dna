import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from typing import Optional

from lstm_cds.model import LSTMModel
from lstm_cds.loader import load_genbank
from lstm_cds.functional import binary_accuracy


def train(
        gbk: str,
        seq_len: int,
        batch_size: int,
        learning_rate: float,
        n_epochs: int,
        in_model_file: Optional[str],
        out_model_file: str,
        training_log_csv: str,
        cuda: bool):

    X, Y = load_genbank(gbks=gbk, seq_len=seq_len, cuda=cuda)

    if in_model_file is not None:
        lstm_model = torch.load(in_model_file)
    else:
        lstm_model = LSTMModel(cuda=cuda)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    df = pd.DataFrame(columns=['epoch', 'sample', 'loss', 'accuracy'])

    n_samples, _, _ = X.size()

    for epoch in range(n_epochs):

        sample = 0
        while sample < n_samples:

            x = X[sample:sample+batch_size, :, :]
            y = Y[sample:sample+batch_size, :, :]

            optimizer.zero_grad()

            y_hat = lstm_model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            y_hat = torch.sigmoid(y_hat)
            accuracy = binary_accuracy(y_hat, y)

            row = {
                'epoch': epoch,
                'sample': sample,
                'loss': loss.item(),
                'accuracy': accuracy
            }
            df = df.append(row, ignore_index=True)

            sample += batch_size

        df.to_csv(training_log_csv, index=False)

        bool_arr = df['epoch'] == epoch
        subdf = df[bool_arr]
        loss = sum(subdf['loss']) / len(subdf)
        accu = sum(subdf['accuracy']) / len(subdf)

        print(f'epoch={epoch}, loss={loss}, accuracy={accu}')

    torch.save(lstm_model, out_model_file)


def evaluate(
        gbk: str,
        model_file: str,
        seq_len: int,
        batch_size: int,
        cuda: bool):

    X, Y = load_genbank(gbks=gbk, seq_len=seq_len, cuda=cuda)

    print(X.size())

    model = torch.load(model_file)
    criterion = nn.BCEWithLogitsLoss()

    n_samples, _, _ = X.size()

    with torch.no_grad():
        loss = []
        accuracy = []
        sample = 0
        while sample < n_samples:
            x = X[sample:sample + batch_size, :, :]
            y = Y[sample:sample + batch_size, :, :]

            y_hat = model(x)
            loss.append(criterion(y_hat, y))

            y_hat = torch.sigmoid(y_hat)
            accuracy.append(binary_accuracy(y_hat, y))

            sample += batch_size

    loss = sum(loss) / len(loss)
    accuracy = sum(accuracy) / len(accuracy)

    print(f'loss={loss}, accuracy={accuracy}')


if __name__ == '__main__':
    # train(
    #     gbk='./data/Pseudomonas_aeruginosa_UCBPP-PA14_109.gbk',
    #     seq_len=100,
    #     cuda=True,
    #     batch_size=1000,
    #     learning_rate=1e-3,
    #     n_epochs=100,
    #     in_model_file=None,
    #     out_model_file='./lstm_model_1',
    #     training_log_csv='./training_log_1.csv')

    # train(
    #     gbk='./data/Pseudomonas_aeruginosa_UCBPP-PA14_109.gbk',
    #     seq_len=200,
    #     cuda=True,
    #     batch_size=500,
    #     learning_rate=1e-5,
    #     n_epochs=100,
    #     in_model_file='./lstm_model_1',
    #     out_model_file='./lstm_model_2',
    #     training_log_csv='./training_log_2.csv')

    evaluate(
        gbk='./data/Pseudomonas_putida_F1_118.gbk',
        model_file='./lstm_model_2',
        seq_len=1000,
        batch_size=100,
        cuda=True)
