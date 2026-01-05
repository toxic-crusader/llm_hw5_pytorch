# File: scripts/train.py
from typing import List, Tuple, Union

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


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[float, List[int], List[int]]:
    """
    Evaluates the model on the validation dataset.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : nn.Module
        Loss function.
    device : Union[str, torch.device], optional
        Device used for evaluation. Defaults to "cpu".

    Returns
    -------
    Tuple[float, List[int], List[int]]
        Average validation loss, list of true labels, list of predicted labels.
    """

    try:
        from IPython.display import Markdown, display

        def md(text: str):
            display(Markdown(text))
    except Exception:

        def md(text: str):
            pass

    model.eval()

    md("### Validation")
    md(f"Device `{device}`")
    md(f"Number of batches `{len(val_loader)}`")

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(val_loader)

    md("### Validation finished")
    md(f"Average validation loss `{avg_loss:.4f}`")

    return avg_loss, y_true, y_pred
