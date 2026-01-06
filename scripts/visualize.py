# File: scripts/visualize.py

from pathlib import Path
from typing import List, Union
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt


def show_misclassified_samples(
    model: torch.nn.Module,
    data_dir: Union[str, Path],
    class_names: List[str],
    device: Union[str, torch.device] = "cpu",
    max_samples: int = 5,
):
    """
    Displays misclassified validation samples with file paths and predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    data_dir : Union[str, Path]
        Root directory of the processed dataset.
    class_names : List[str]
        List of class names.
    device : Union[str, torch.device], optional
        Device used for inference.
    max_samples : int, optional
        Maximum number of misclassified samples to display.
    """

    data_dir = Path(data_dir).resolve()
    val_dir = data_dir / "val"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ImageFolder(root=val_dir, transform=transform)

    model.eval()
    shown = 0

    with torch.no_grad():
        for idx, (image, label) in enumerate(dataset):
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1).item()

            if pred != label:
                image_path = dataset.samples[idx][0]

                img = plt.imread(image_path)
                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis("off")
                plt.title(
                    f"True: {class_names[label]} | Predicted: {class_names[pred]}"
                )
                plt.show()

                print(f"Image path: {image_path}")

                shown += 1
                if shown >= max_samples:
                    break
