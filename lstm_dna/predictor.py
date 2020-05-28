import torch
from typing import List
from lstm_dna.model import LSTMModel
from lstm_dna.loader import dna_to_one_hot


class Predictor:

    def __init__(
            self,
            model: LSTMModel,
            window: int = 1024,
            mini_batch_size: int = 32):
        """
        Args:
            model

            window:
                Input sequence length for LSTM model
                Each window is a "sample"

            mini_batch_size
        """

        self.model = model
        self.window = window
        self.mini_batch_size = mini_batch_size

    def divide_windows(
            self,
            X: torch.tensor) -> torch.Tensor:
        """
        Break full_len -> n_samples * self.window

        Args:
            X: 3D tensor, size (1, full_len, 4), dtype=float

        Returns:
            X: 3D tensor, size (n_samples, self.window, 4), dtype=float
        """
        n, full_len, input_size = X.size()

        assert n == 1
        assert input_size == 4

        quotient = full_len // self.window
        remainder = full_len % self.window

        if remainder == 0:
            n_samples = quotient
        else:
            n_samples = quotient + 1
            pad_len = self.window - remainder

            X_pad = torch.zeros(1, pad_len, input_size, dtype=torch.float)
            if X.is_cuda:
                X_pad = X_pad.cuda()

            X = torch.cat([X, X_pad], dim=1)

        X = X.view(n_samples, self.window, 4)

        return X

    def forward_pass(
            self,
            X: torch.Tensor) -> torch.Tensor:
        """
        1) Break the input X into segments, or windows
        2) Forward pass to get Y, i.e. predicted probability
        3) Flatten windows of Y back to full input length

        Flat-in, flat-out

        Args:
            X: 3D tensor, size (1, full_len, 4), dtype=float

        Returns:
            Y: 3D tensor, size (1, full_len, 1), dtype=float
        """

        n, full_len, _ = X.size()

        assert n == 1

        X = self.divide_windows(X=X)

        n_samples, window, _ = X.size()

        assert window == self.window

        Y = torch.zeros(n_samples, window, 1)

        i = 0
        mb = self.mini_batch_size
        while i < n_samples:
            start = i
            end = min(i + mb, n_samples)  # the last mini_batch < mini_batch_size

            x = X[start:end, :, :]

            if self.model.is_cuda:
                x = x.cuda()

            with torch.no_grad():
                y = torch.sigmoid(self.model(x))  # sigmoid probability

            Y[start:end, :, :] = y[:, :, :]

            i += mb

        Y = Y.view(1, -1, 1)  # flatten -> (1, full_len)
        Y = Y[:, 0:full_len, :]  # remove padded sequence

        return Y

    def predict(
            self,
            dna: str,
            coverage: int = 4) -> List[bool]:

        full_len = len(dna)

        X = dna_to_one_hot(seq=dna)
        assert X.size() == (full_len, 4)

        X = X.view(1, -1, 4)
        assert X.size() == (1, full_len, 4)

        Y_sum = torch.zeros(1, full_len, 1)
        Y_count = torch.zeros(1, full_len, 1)

        offset: int = self.window // coverage
        for i in range(coverage):
            start = i * offset
            Y_sum[:, start:, :] += self.forward_pass(X=X[:, start:, :])
            Y_count[:, start:, :] += 1

        Y = Y_sum / Y_count
        Y = Y.view(-1)
        prediction = (Y > 0.5).tolist()

        return prediction
