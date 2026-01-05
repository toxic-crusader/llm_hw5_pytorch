# File: scripts/metrics.py
from typing import List

from IPython.display import Markdown, display
from sklearn.metrics import classification_report


def classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> str:
    """
    Computes and displays classification metrics for each class.

    Parameters
    ----------
    y_true : List[int]
        Ground truth labels.
    y_pred : List[int]
        Predicted labels.
    class_names : List[str]
        List of class names corresponding to label indices.

    Returns
    -------
    str
        Text classification report.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    display(Markdown("### Classification metrics"))
    display(Markdown("```\n" + report + "\n```"))

    return report
