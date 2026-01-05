# File: scripts/train.py
from typing import Union
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device] = "cpu",
) -> float:
    """
    Trains the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    device : Union[str, torch.device], optional
        Device used for training. Defaults to "cpu".

    Returns
    -------
    float
        Average training loss for the epoch.
    """

    try:
        from IPython.display import Markdown, display

        def md(text: str):
            display(Markdown(text))
    except Exception:

        def md(text: str):
            pass

    model.train()

    md("### Training one epoch")
    md(f"Device `{device}`")
    md(f"Number of batches `{len(train_loader)}`")

    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    md("### Epoch finished")
    md(f"Average training loss `{avg_loss:.4f}`")

    return avg_loss
