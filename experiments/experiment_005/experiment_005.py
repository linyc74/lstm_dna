import os
import torch
import torch.nn as nn
import torch.optim as optim
from ngslite import get_files
from lstm_dna.trainer import Trainer
from lstm_dna.model import LSTMModel
from lstm_dna.loader import load_genbank, divide_sequence, split, shuffle
from lstm_dna.functional import softmax_accuracy, binary_accuracy, get_class_weight


EXPERIMENT_NAME = f'{os.path.basename(__file__)[:-3]}'


# Data
GBKDIR = '../data'
LABEL_LENGTH = None  # full CDS


# Model
ARCHITECTURES = [
    [4, 128, 64, 2],
    [4, 128, 2],
    [4, 64, 2],
]
BIDIRECTIONAL = True
SOFTMAX = True  # False: binary classification
CUDA = True


# Training
TRAINING_STAGES = [
    (1, 128, 128, 1e-3),  # stage, mini_batch_size, seq_len, learning_rate
    (2, 16, 1024, 1e-4),
]
TRAINING_FRACTION = 0.7
MAX_EPOCHS = 1000
OVERTRAIN_EPOCHS = 5


def train_model(
        gbkdir: str,
        stage: int,
        mini_batch_size: int,
        seq_len: int,
        learning_rate: float,
        model: LSTMModel,
        model_name: str) -> LSTMModel:

    gbks = get_files(source=gbkdir, isfullpath=True)
    X, Y = load_genbank(gbks=gbks, label_length=LABEL_LENGTH)

    if not SOFTMAX:  # Reshape for binary classification
        Y = Y.view(-1, 1).float()  # 1D long -> 2D float

    X = divide_sequence(X, seq_len=seq_len, pad=True)
    Y = divide_sequence(Y, seq_len=seq_len, pad=True)

    X, Y = shuffle(X, Y)

    X_train, X_test = split(X, training_fraction=TRAINING_FRACTION, dim=0)
    Y_train, Y_test = split(Y, training_fraction=TRAINING_FRACTION, dim=0)

    weight = get_class_weight(Y)
    if CUDA:
        weight = weight.cuda()

    criterion = nn.CrossEntropyLoss(weight=weight) if SOFTMAX else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log_dir = f'tensorboard/{EXPERIMENT_NAME}/{model_name}_stage_{stage}'
    accuracy = softmax_accuracy if SOFTMAX else binary_accuracy

    trainer = Trainer(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        criterion=criterion,
        optimizer=optimizer,
        accuracy=accuracy,
        mini_batch_size=mini_batch_size,
        log_dir=log_dir)

    model = trainer.train(
        max_epochs=MAX_EPOCHS,
        overtrain_epochs=OVERTRAIN_EPOCHS)

    return model


def main():

    for i, architecture in enumerate(ARCHITECTURES):

        model = LSTMModel(
            sizes=architecture,
            bidirectional=BIDIRECTIONAL,
            softmax_output=SOFTMAX,
            cuda=CUDA)

        for stage, mini_batch_size, seq_len, learning_rate in TRAINING_STAGES:

            model = train_model(
                gbkdir=GBKDIR,
                stage=stage,
                mini_batch_size=mini_batch_size,
                seq_len=seq_len,
                learning_rate=learning_rate,
                model=model,
                model_name=f'model_{i+1}')

            os.makedirs('./models', exist_ok=True)
            torch.save(model, f'./models/{EXPERIMENT_NAME}_model_{i+1}_stage_{stage}.model')

        torch.save(model, f'./models/{EXPERIMENT_NAME}_model_{i+1}.model')


if __name__ == '__main__':
    main()
