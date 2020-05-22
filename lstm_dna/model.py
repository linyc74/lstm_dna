import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LSTMModel(nn.Module):

    def __init__(
            self,
            sizes: List[int],
            batch_first: bool,
            bidirectional: bool,
            output_activation: Optional[str] = None,
            cuda: bool = False):
        """
        sizes: [input_size, hidden_size_1, hidden_size_2, ..., output_size]
        """

        super().__init__()

        assert len(sizes) >= 3
        assert output_activation in [None, 'sigmoid', 'softmax']

        self.output_activation = output_activation
        self.n_layers = len(sizes) - 2  # number of LSTM layers

        for i in range(len(sizes) - 2):

            input_size = sizes[i]
            hidden_size = sizes[i+1]

            if i > 0 and bidirectional:
                input_size *= 2

            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=batch_first,
                bidirectional=bidirectional)

            setattr(self, f'lstm_{i+1}', lstm)

        in_features = sizes[-2]
        out_features = sizes[-1]

        if bidirectional:
            in_features *= 2

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features)

        if cuda:
            self.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n_samples, seq_len, input_size)

        Return:
            y_hat: (n_samples, seq_len, output_size)
        """
        for i in range(self.n_layers):
            lstm = getattr(self, f'lstm_{i+1}')
            x, _ = lstm(x)

        x = self.fc(x)

        if self.output_activation is None:
            y_hat = x
        elif self.output_activation == 'sigmoid':
            y_hat = F.sigmoid(x)
        else:  # softmax
            y_hat = F.softmax(x, dim=2)

        return y_hat
