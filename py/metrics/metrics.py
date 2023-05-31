import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


def accuracy(confusion_matrix):
    """Calculate the accuracy of a confusion matrix via the sum of the diagonal."""
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


def acc_score(y_true, y_pred):
    """Calculate the accuracy score."""
    return accuracy_score(y_true, y_pred)


def report(y_true, y_pred, labels):
    """Print a classification report."""
    cls_report: str | dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    return pd.DataFrame(cls_report).transpose()
