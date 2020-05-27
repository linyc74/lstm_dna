import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, List
from .model import LSTMModel
from .functional import binary_accuracy


class Trainer:
    """
    Automatically trains a model until it gets the minimum test loss

    The minimum test loss is deterined by N consecutive epochs with higher losses
        after the minimum test loss is achieved,
        where N = <overtrain_epochs>

    The (best) model with minimum test loss is returned by the .train() method
    """

    epoch: int = 0
    min_test_loss: Optional[float] = None
    overtrained_losses: List[float] = []
    best_model_state_dict: dict = {}

    # min_test_loss:
    #     The minimum test loss achieved so far
    # overtrained_losses:
    #     A list of losses that are greater than min_test_loss, and take place after min_test_loss

    def __init__(
            self,
            model: LSTMModel,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            X_test: torch.Tensor,
            Y_test: torch.Tensor,
            criterion,
            optimizer,
            mini_batch_size: int,
            writer: SummaryWriter):
        """
        X_train: (n_training_samples, seq_len, 4)
        Y_train: (n_training_samples, seq_len, 1)
        X_test: (n_test_samples, seq_len, 4)
        Y_test: (n_test_samples, seq_len, 1)
        """

        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.criterion = criterion
        self.optimizer = optimizer
        self.mini_batch_size = mini_batch_size
        self.writer = writer

    def train_one_epoch(self):
        mb = self.mini_batch_size
        n_samples, _, _ = self.X_train.size()

        i = 0
        while i < n_samples:
            start = i
            end = min(i + mb, n_samples)  # the last mini_batch < mini_batch_size

            x = self.X_train[start:end, :, :]
            y = self.Y_train[start:end, :, :]

            if self.model.is_cuda:
                x = x.cuda()
                y = y.cuda()

            self.optimizer.zero_grad()

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            i += mb

    def tensorboard(
            self,
            tag: str,
            loss: float,
            accuracy: float):

        self.writer.add_scalar(
            tag=f'Loss ({tag})', scalar_value=loss, global_step=self.epoch)

        self.writer.add_scalar(
            tag=f'Accuracy ({tag})', scalar_value=accuracy, global_step=self.epoch)

    def validate(self) -> float:

        mb = self.mini_batch_size
        scalars = {}

        with torch.no_grad():
            for set_ in ['train', 'test']:
                X = getattr(self, f'X_{set_}')
                Y = getattr(self, f'Y_{set_}')

                n_samples, _, _ = X.size()

                i = 0
                total_loss = 0
                total_accuracy = 0
                while i < n_samples:
                    start = i
                    end = min(i + mb, n_samples)  # the last mini_batch < mini_batch_size

                    x = X[start:end, :, :]
                    y = Y[start:end, :, :]

                    if self.model.is_cuda:
                        x = x.cuda()
                        y = y.cuda()

                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)
                    accuracy = binary_accuracy(torch.sigmoid(y_hat), y)

                    total_loss += loss.item() * (end - start)
                    total_accuracy += accuracy * (end - start)

                    i += mb

                scalars[f'{set_}_loss'] = total_loss / n_samples
                scalars[f'{set_}_accuracy'] = total_accuracy / n_samples

                self.tensorboard(
                    tag=set_,
                    loss=scalars[f'{set_}_loss'],
                    accuracy=scalars[f'{set_}_accuracy']
                )

        return scalars['test_loss']

    def decide(
            self,
            test_loss: float,
            overtrain_epochs: int) -> bool:

        if self.min_test_loss is None:
            self.min_test_loss = test_loss
            self.best_model_state_dict = self.model.state_dict()
            return False

        overtrained = test_loss > self.min_test_loss

        if overtrained:
            self.overtrained_losses.append(test_loss)
        else:
            self.min_test_loss = test_loss
            self.best_model_state_dict = self.model.state_dict()
            self.overtrained_losses = []  # reset overtrained losses to empty

        if len(self.overtrained_losses) == overtrain_epochs:
            return True
        else:
            return False

    def train(
            self,
            max_epochs: int,
            overtrain_epochs: int):

        while self.epoch < max_epochs:

            self.train_one_epoch()

            test_loss = self.validate()

            complete = self.decide(
                test_loss=test_loss,
                overtrain_epochs=overtrain_epochs)

            if complete:
                break

            self.epoch += 1

        self.model.load_state_dict(self.best_model_state_dict)
        return self.model
