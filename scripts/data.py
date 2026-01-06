# File: scripts/data.py
from pathlib import Path
from typing import List, Tuple, Union

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 2,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creates PyTorch DataLoaders from a preprocessed image dataset.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Root directory of the preprocessed dataset.
        The directory must contain `train` and `val` subdirectories.
    batch_size : int, optional
        Number of samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of worker processes for data loading. Defaults to 2.
    augment : bool, optional
        If True, apply data augmentation to the training dataset only.
        Defaults to False.

    Returns
    -------
    Tuple[DataLoader, DataLoader, List[str]]
        Train DataLoader, validation DataLoader and list of class names.
    """

    try:
        from IPython.display import Markdown, display

        def md(text: str):
            display(Markdown(text))
    except Exception:
        def md(text: str):
            pass

    data_dir = Path(data_dir).resolve()
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    md("### Loading processed dataset")
    md(f"Dataset directory `{data_dir}`")
    md(f"Batch size `{batch_size}`")
    md(f"Augmentation `{augment}`")

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        train_transform = val_transform

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_names = train_dataset.classes

    md("### Dataset loaded")
    md(f"Number of classes `{len(class_names)}`")
    md(f"Class names `{class_names}`")
    md(f"Train samples `{len(train_dataset)}`")
    md(f"Validation samples `{len(val_dataset)}`")

    return train_loader, val_loader, class_names
