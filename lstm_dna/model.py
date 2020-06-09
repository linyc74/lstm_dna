import torch
import torch.nn as nn
from typing import List


BATCH_FIRST = True


class LSTMModel(nn.Module):

    def __init__(
            self,
            sizes: List[int],
            bidirectional: bool,
            softmax_output: bool,
            cuda: bool):
        """
        sizes: [input_size, hidden_size_1, hidden_size_2, ..., output_size]
        """
        super().__init__()

        assert len(sizes) >= 3

        self.lstm_layers = nn.ModuleList()

        for i in range(len(sizes) - 2):

            input_size, hidden_size = sizes[i:i+2]

            if i > 0 and bidirectional:
                input_size *= 2

            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=BATCH_FIRST,
                bidirectional=bidirectional)

            self.lstm_layers.append(lstm)

        in_features, out_features = sizes[-2:]

        if bidirectional:
            in_features *= 2

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features)

        if cuda:
            self.cuda()
        self.is_cuda = cuda

        self.softmax_output = softmax_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                (batch_size, seq_len, input_size)

        Return:
            y_hat:
                (batch_size, seq_len, output_size)

                If self.softmax_output = True, (batch_size, output_size, seq_len)
        """
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)

        y_hat = self.fc(x)

        if self.softmax_output:
            y_hat = y_hat.transpose(1, 2)

        return y_hat
