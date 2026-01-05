# File: scripts/preprocess.py
import os
import shutil
from pathlib import Path
from typing import Union
from PIL import Image


def preprocess_dataset(
    raw_dir: Union[str, Path],
    processed_dir: Union[str, Path],
    image_size: int = 224,
    force: bool = False,
) -> Path:
    """
    Preprocesses a dataset of images and saves the result to a target directory.

    If the target directory already exists and force is False, the function
    skips preprocessing and returns the existing path.

    Parameters
    ----------
    raw_dir : Union[str, Path]
        Source directory of the raw dataset.
    processed_dir : Union[str, Path]
        Target directory for the preprocessed dataset.
    image_size : int, optional
        Final image size after preprocessing. Defaults to 224.
    force : bool, optional
        If True, existing processed data will be removed and regenerated.
        Defaults to False.

    Returns
    -------
    Path
        Path object pointing to the root of the preprocessed dataset.
    """

    try:
        from IPython.display import Markdown, display

        def md(text: str):
            
            display(Markdown(text))
    except Exception:

        def md(text: str):
            pass

    raw_dir = Path(raw_dir).resolve()
    processed_dir = Path(processed_dir).resolve()

    md("### Preprocessing dataset")
    md(f"Source directory `{raw_dir}`")
    md(f"Target directory `{processed_dir}`")
    md(f"Image size `{image_size} x {image_size}`")
    md(f"Force reprocessing `{force}`")

    if processed_dir.exists() and any(processed_dir.iterdir()) and not force:
        md("### Preprocessing skipped")
        md("Processed data already exists on disk")
        return processed_dir

    if processed_dir.exists() and force:
        shutil.rmtree(processed_dir)

    total_images = 0
    skipped_images = 0

    for root, _, files in os.walk(raw_dir):
        root_path = Path(root)
        rel_path = root_path.relative_to(raw_dir)
        target_root = processed_dir / rel_path
        target_root.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_path = root_path / file
            dst_path = target_root / file

            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    img = img.resize((image_size, image_size))
                    img.save(dst_path)
                    total_images += 1
            except Exception:
                skipped_images += 1
                continue

    md("### Preprocessing finished")
    md(f"Processed images `{total_images}`")
    md(f"Skipped images `{skipped_images}`")

    return processed_dir
