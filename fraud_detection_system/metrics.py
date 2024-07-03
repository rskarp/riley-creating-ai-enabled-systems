import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import Model


class Metrics:
    """
    Metrics class for evaluating binary classification models.
    """

    def __init__(self, data: pd.DataFrame, model_version: str):
        """
        Initialize the Metrics class with a model and dataset.

        Parameters:
        data (DataFrame): the dataset of features.
        model_version (str): the name of the model version to use for inference.
        """
        self.model = Model()
        try:
            self.model.set_model(f'resources/models/{model_version}.joblib')
        except FileNotFoundError:
            raise FileNotFoundError(f'Model not {model_version} found')
        self.dataset = data
        self.y_true = self.dataset.loc[:, 'is_fraud']
        self.predicted_labels = self.evaluate()

    def evaluate(self):
        """
        Evaluate model using the specified dataset version and its label column.

        Returns:
        ndarray: The predicted labels for self.dataset using self.model.
        """
        unlabeled = self.dataset.loc[:, self.dataset.columns != 'is_fraud']
        return self.model.predict(unlabeled)

    def accuracy(self):
        """
        Calculate the accuracy of the model.

        Returns:
        float: The accuracy score.
        """
        return accuracy_score(self.y_true, self.predicted_labels)

    def precision(self):
        """
        Calculate the precision of the model.

        Returns:
        float: The precision score.
        """
        return precision_score(self.y_true, self.predicted_labels)

    def recall(self):
        """
        Calculate the recall of the model.

        Returns:
        float: The recall score.
        """
        return recall_score(self.y_true, self.predicted_labels)

    def f1(self):
        """
        Calculate the F1 score of the model.

        Returns:
        float: The F1 score.
        """
        return f1_score(self.y_true, self.predicted_labels)

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
