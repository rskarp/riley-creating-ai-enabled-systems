import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import Model


class Metrics:
    """
    Metrics class for evaluating binary classification models.
    """

    def __init__(self):
        """
        Initialize the Metrics class with a model and dataset.
        """
        self.model = Model()

    def evaluate(self):
        """
        Evaluate model using the specified dataset version and its label column.

        """
        pass

    def accuracy(self):
        """
        Calculate the accuracy of the model.

        Returns:
        float: The accuracy score.
        """
        return accuracy_score(self.y_eval, self.predicted_labels)

    def precision(self):
        """
        Calculate the precision of the model.

        Returns:
        float: The precision score.
        """
        return precision_score(self.y_eval, self.predicted_labels)

    def recall(self):
        """
        Calculate the recall of the model.

        Returns:
        float: The recall score.
        """
        return recall_score(self.y_eval, self.predicted_labels)

    def f1(self):
        """
        Calculate the F1 score of the model.

        Returns:
        float: The F1 score.
        """
        return f1_score(self.y_eval, self.predicted_labels)

    def run_metrics(self):
        """
        Calculate all the metrics: accuracy, precision, recall, and F1 score.

        Returns:
        dict: A dictionary containing all the metrics.
        """
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1()
        }
