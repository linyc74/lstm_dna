import torch


def binary_accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    y_hat = (y_hat > 0.5).float()
    correct = (y_hat == y).float().sum().item()
    accuracy = correct / y.nelement()
    return accuracy
