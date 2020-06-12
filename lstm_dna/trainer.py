import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, List, Dict, Tuple
from .model import LSTMModel


class Trainer:
    """
    Automatically trains a model until it gets the minimum experiment_007 loss

    The minimum experiment_007 loss is determined by N consecutive epochs
        after the minimum experiment_007 loss is achieved, where N = <overtrain_epochs>

    The (best) model with minimum experiment_007 loss is returned by the .train() method
    """

    epoch: int = 0
    min_test_loss: Optional[float] = None  # The minimum experiment_007 loss achieved so far
    overtrained_losses: List[float] = []   # Higher loss values that are after min_test_loss
    best_model_state_dict: dict = {}

    def __init__(
            self,
            model: LSTMModel,
            X_train: torch.Tensor,
            Y_train: torch.Tensor,
            X_test: torch.Tensor,
            Y_test: torch.Tensor,
            criterion,
            optimizer,
            accuracy,
            mini_batch_size: int,
            log_dir: str):
        """
        X_train: (n_samples, ...)
        Y_train: (n_samples, ...)
        X_test: (n_samples, ...)
        Y_test: (n_samples, ...)
        """

        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.criterion = criterion
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.mini_batch_size = mini_batch_size
        self.log_dir = log_dir

    def __train_one_epoch(self):
        mb = self.mini_batch_size
        n_samples = self.X_train.size()[0]

        i = 0
        while i < n_samples:
            start = i
            end = min(i + mb, n_samples)  # the last mini batch < mini_batch_size

            x = self.X_train[start:end]
            y = self.Y_train[start:end]

            if self.model.is_cuda:
                x = x.cuda()
                y = y.cuda()

            self.optimizer.zero_grad()

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            i += mb

    def __tensorboard(
            self,
            writer: SummaryWriter,
            values: Dict[str, float]):

        for key, val in values.items():
            tag = key.replace('_', ' ').title()
            writer.add_scalar(
                tag=tag, scalar_value=val, global_step=self.epoch)

    def __decide(
            self,
            test_loss: float,
            overtrain_epochs: int) -> bool:
        """
        Returns complete or not
        """

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

    def __compute_loss_accuracy(
            self,
            X: torch.Tensor,
            Y: torch.Tensor) -> Tuple[float, float]:

        mb = self.mini_batch_size
        n_samples = X.size()[0]

        i = 0
        total_loss, total_accuracy = 0, 0
        while i < n_samples:
            start = i
            end = min(i + mb, n_samples)  # the last mini batch < mini_batch_size

            x = X[start:end]
            y = Y[start:end]

            if self.model.is_cuda:
                x = x.cuda()
                y = y.cuda()

            with torch.no_grad():
                y_hat = self.model(x)

            loss = self.criterion(y_hat, y)
            accuracy = self.accuracy(y_hat, y)

            total_loss += loss.item() * (end - start)
            total_accuracy += accuracy * (end - start)

            i += mb

        avg_loss = total_loss / n_samples
        avg_accuracy = total_accuracy / n_samples

        return avg_loss, avg_accuracy

    def validate(self) -> Dict[str, float]:

        values = {}

        for set_ in ['train', 'experiment_007']:
            X = getattr(self, f'X_{set_}')
            Y = getattr(self, f'Y_{set_}')

            loss, accuracy = self.__compute_loss_accuracy(X=X, Y=Y)

            values[f'{set_}_loss'] = loss
            values[f'{set_}_accuracy'] = accuracy

        return values

    def train(
            self,
            max_epochs: int,
            overtrain_epochs: int):

        writer = SummaryWriter(log_dir=self.log_dir)

        while self.epoch < max_epochs:

            self.__train_one_epoch()
            values = self.validate()
            self.__tensorboard(writer=writer, values=values)

            complete = self.__decide(
                test_loss=values['test_loss'],
                overtrain_epochs=overtrain_epochs)
            if complete:
                break

            self.epoch += 1

        writer.close()

        self.model.load_state_dict(self.best_model_state_dict)
        return self.model
