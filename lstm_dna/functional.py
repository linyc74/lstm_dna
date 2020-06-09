import torch


def binary_accuracy(
        y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    y_hat and y should have the same shape

    Args:
        y_hat: probability
        y: 0 or 1
    """

    y_hat = (y_hat > 0.5).float()
    correct = (y_hat == y).float().sum().item()
    accuracy = correct / y.nelement()

    return accuracy


def softmax_accuracy(
        y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    y_hat has an extra dimension compared to y
    The extra dimension C is always the second one

    Args:
        y_hat: (N, C, dims, ...)
        y: (N, dims)
    """

    y_hat = y_hat.argmax(dim=1)  # collapse the second dim
    correct = (y_hat == y).float().sum().item()
    accuracy = correct / y.nelement()

    return accuracy


def get_class_weight(
        Y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        Y: any shape, dtype=torch.long, values = 0, ..., C-1

    Returns:
        weight: 1D tensor, (C, ), dtype=torch.float
    """

    n = Y.nelement()
    counts = []
    n_classes = Y.max().item() + 1
    for c in range(n_classes):
        count = (Y == c).int().sum().item()
        counts.append(count)

    counts = torch.tensor(counts, dtype=torch.float)

    weight = n / counts
    weight = weight / weight.sum()

    return weight
