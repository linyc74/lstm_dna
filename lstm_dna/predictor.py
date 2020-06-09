import torch
from typing import List, Callable
from lstm_dna.model import LSTMModel
from lstm_dna.loader import dna_to_one_hot, divide_sequence


def binary_output_to_label(y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        y: 3D tensor (batch_size, seq_len, output_size)
           output_size = 1

    Returns:
        y: 2D tensor (batch_size, seq_len)
    """
    n_samples, seq_len, output_size = y.shape
    assert output_size == 1
    return y.view(n_samples, seq_len)


def softmax_output_to_label(y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        y: 3D tensor (batch_size, output_size, seq_len)
           output_size = n_classes

    Returns:
        y: 2D tensor (batch_size, seq_len)
    """
    return y.argmax(dim=1)


class Predictor:

    def __init__(
            self,
            model: LSTMModel,
            output_to_label: Callable,
            window: int = 1024,
            coverage: int = 4,
            mini_batch_size: int = 32):
        """
        Args:
            model

            output_to_label:
                A callable function that converts the model output (3D tensor) to label (2D tensor)
                2D tensor: (batch_size, seq_len)
                The model output depends on the type or prediction, binary or softmax

            window:
                Input sequence length for LSTM model
                Each window is a "sample"

            mini_batch_size
        """

        self.model = model
        self.output_to_label = output_to_label
        self.window = window
        self.coverage = coverage
        self.mini_batch_size = mini_batch_size

    def __forward_pass(
            self,
            X: torch.Tensor) -> torch.Tensor:
        """
        1) Break the input X into segments, or windows
        2) Forward pass to get Y, i.e. predicted probability
        3) Flatten windows of Y back to full input length

        Flat-in, flat-out

        Args:
            X: 2D tensor, size (full_len, 4), dtype=torch.float

        Returns:
            Y: 1D tensor, size (full_len, ), dtype=torch.float
        """

        full_len = X.shape[0]

        X = divide_sequence(X, seq_len=self.window, pad=True)  # 2D -> 3D

        n_samples, window, _ = X.size()

        Y = torch.zeros(n_samples, window)  # 2D

        i = 0
        mb = self.mini_batch_size
        while i < n_samples:
            start = i
            end = min(i + mb, n_samples)  # the last mini_batch < mini_batch_size

            x = X[start:end, :, :]

            if self.model.is_cuda:
                x = x.cuda()

            with torch.no_grad():
                y_hat = self.model(x)
                y = self.output_to_label(y_hat)  # 3D -> 2D tensor (batch_size, window)

            Y[start:end, :] = y[:, :]

            i += mb

        Y = Y.view(-1)  # 2D (n_samples, seq_len) -> 1D (full_len)
        Y = Y[0:full_len]  # remove padded sequence

        return Y.float()

    def predict(self, dna: str) -> List[bool]:
        """
        Args:
            dna: DNA sequence

        Returns:
            List of bool, length = len(dna)
        """

        full_len = len(dna)

        X = dna_to_one_hot(seq=dna)
        assert X.size() == (full_len, 4)

        Y_sum = torch.zeros(full_len)
        Y_count = torch.zeros(full_len)

        shift: int = self.window // self.coverage
        for i in range(self.coverage):
            start = i * shift
            Y_sum[start:] += self.__forward_pass(X=X[start:])
            Y_count[start:] += 1

        Y = Y_sum / Y_count
        Y = Y.view(-1)
        prediction = (Y > 0.5).tolist()

        return prediction
