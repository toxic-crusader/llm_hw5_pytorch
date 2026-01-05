# File: scripts/model.py
from typing import Union
import torch
from torch import nn
from torchvision import models


def build_model(
    num_classes: int = 3,
    freeze_backbone: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Builds a ResNet18-based classification model.

    Parameters
    ----------
    num_classes : int, optional
        Number of target classes. Defaults to 3.
    freeze_backbone : bool, optional
        If True, all backbone layers are frozen and only the classifier
        head is trained. Defaults to True.
    device : Union[str, torch.device], optional
        Device to move the model to. Defaults to "cpu".

    Returns
    -------
    nn.Module
        PyTorch model ready for training or inference.
    """

    try:
        from IPython.display import Markdown, display

        def md(text: str):
            display(Markdown(text))
    except Exception:

        def md(text: str):
            pass

    md("### Building model")
    md("Architecture `ResNet18`")
    md(f"Number of classes `{num_classes}`")
    md(f"Freeze backbone `{freeze_backbone}`")
    md(f"Device `{device}`")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)

    md("### Model ready")
    md(f"Classifier in_features `{in_features}`")
    md(f"Classifier out_features `{num_classes}`")

    return model
