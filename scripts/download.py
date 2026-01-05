# File: scripts/download.py
import shutil
from pathlib import Path
from typing import Union
import kagglehub


def download_dataset(
    dataset_id: str,
    target_dir: Union[str, Path],
    force: bool = False,
) -> Path:
    """
    Downloads a dataset from Kaggle and copies it to the specified target directory.

    If the target directory already exists and is not empty, the download
    is skipped unless force is set to True.

    Parameters
    ----------
    dataset_id : str
        The ID of the Kaggle dataset to download.
    target_dir : Union[str, Path]
        Target directory where the dataset should be stored.
    force : bool, optional
        If True, existing data will be removed and re downloaded.
        Defaults to False.

    Returns
    -------
    Path
        Path object pointing to the local dataset directory.
    """

    try:
        from IPython.display import Markdown, display

        def md(text: str):
            display(Markdown(text))
    except Exception:

        def md(text: str):
            pass

    target_dir = Path(target_dir).resolve()

    md("### Downloading dataset")
    md(f"Dataset id `{dataset_id}`")
    md(f"Target directory `{target_dir}`")
    md(f"Force download `{force}`")

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        md("### Download skipped")
        md("Dataset already exists locally")
        return target_dir

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    source_path = Path(kagglehub.dataset_download(dataset_id)).resolve()

    shutil.copytree(source_path, target_dir)

    md("### Dataset downloaded")
    md(f"Files copied to `{target_dir}`")

    return target_dir
