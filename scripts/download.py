# File: scripts/download.py
import kagglehub
import shutil
import os


def download_dataset(dataset_id: str, target_dir: str) -> str:
    """
    Downloads a dataset from Kaggle and copies it to the specified target directory.

    Args:
        dataset_id (str): The ID of the dataset to download.
        target_dir (str): The directory to copy the dataset to.

    Returns:
        str: The path to the copied dataset.
    """

    try:
        from IPython.display import Markdown, display

        def md(text):
            display(Markdown(text))
    except Exception:

        def md(text):
            pass

    source_path = kagglehub.dataset_download(dataset_id)

    target_dir = os.path.abspath(target_dir)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    shutil.copytree(source_path, target_dir)

    md("### Dataset downloaded")
    md(f"Files copied to `{target_dir}`")

    return target_dir
