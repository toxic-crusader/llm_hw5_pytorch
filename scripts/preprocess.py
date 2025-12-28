# File: scripts/preprocess.py
import os
import shutil
from PIL import Image


def preprocess_dataset(
    raw_dir: str,
    processed_dir: str,
    image_size: int = 224,
) -> str:
    """
    Preprocesses a dataset of images.

    Parameters
    ----------
    raw_dir : str
        Source directory of the dataset.
    processed_dir : str
        Target directory of the preprocessed dataset.
    image_size : int, optional
        Size of the images after preprocessing. Defaults to 224.

    Returns
    -------
    str
        The path to the preprocessed dataset.
    """

    try:
        from IPython.display import Markdown, display

        def md(text: str):
            display(Markdown(text))
    except Exception:

        def md(text: str):
            pass

    raw_dir = os.path.abspath(raw_dir)
    processed_dir = os.path.abspath(processed_dir)

    md("### Preprocessing dataset")
    md(f"Source directory `{raw_dir}`")
    md(f"Target directory `{processed_dir}`")
    md(f"Image size `{image_size} x {image_size}`")

    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    total_images = 0
    skipped_images = 0

    for root, _, files in os.walk(raw_dir):
        rel_path = os.path.relpath(root, raw_dir)
        target_root = os.path.join(processed_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_root, file)

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
