import os
import json
from typing import Dict
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, precision_score, accuracy_score, recall_score, classification_report


class Model:
    """
    Model class that implement model behaviors including 
    versioning, training, and making predictions.
    """

    def __init__(self):
        """
        Initialize the Model with a specified architecture.

        """
        self.model = None
        self.scoring = self.init_model_scoring()

    def init_model_scoring(self) -> Dict:
        '''
        Initialize the scoring metrics used for hyperparameter tuning.

        Parameters:
        None

        Returns:
        Dict:
            The scoring metrics used during hyperparameter tuning
        '''
        ftwo_scorer = make_scorer(fbeta_score, beta=2)
        precision_scorer = make_scorer(precision_score, zero_division=0)
        return {"precision": precision_scorer,
                "recall": "recall", "ftwo_scorer": ftwo_scorer}

    def set_model(self, path: str):
        """
        Load a pre-trained model from a specified path.

        Parameters:
        """
        self.model = joblib.load(path)

    def save_model(self, path: str):
        """
        Load a pre-trained model from a specified path.

        Parameters:
        """
        joblib.dump(self.model, path)

    def train(self, model_version: str, model_type: str, dataset_version: str, hyperparameters: Dict, random_state: int = 1):
        """
        Train the model with provided training data.

        """
        # Read in training data from dataset file
        train_features = pd.read_parquet(
            f'resources/features/{dataset_version}.parquet', engine='pyarrow')
        X_train = train_features.loc[:, train_features.columns != 'is_fraud']
        y_train = train_features.loc[:, 'is_fraud']

        # Determine classifier based on model_type
        if model_type.lower() == 'random_forest':
            clf = RandomForestClassifier(random_state=random_state)
        elif model_type.lower() == 'stochastic_gradient_descent':
            clf = SGDClassifier(random_state=random_state,
                                early_stopping=True, max_iter=10000)
        elif model_type.lower() == 'logistic_regression':
            clf = LogisticRegression(random_state=random_state, max_iter=10000)
        else:
            raise ValueError(f"Model type not allowed: {model_type}")

        # Setup the grid search for hyperparameter tuning
        grid = GridSearchCV(estimator=clf,
                            param_grid=hyperparameters,
                            scoring=self.scoring,
                            refit='ftwo_scorer',
                            cv=5,
                            n_jobs=-1,
                            return_train_score=True,
                            verbose=1)

        # Fit the grid search to the data
        grid.fit(X_train, y_train)

        # Set the model to be the best estimator
        self.model = grid.best_estimator_
        # Fit the best model
        self.model = self.model.fit(X_train, y_train)

        # Save the trained model
        os.makedirs('resources/models', exist_ok=True)
        self.save_model(f'resources/models/{model_version}.joblib')

        # Return description
        description = {
            'model_type': model_type,
            'train_dataset': dataset_version,
            'best_hyperparameters': grid.best_params_,
            'best_ftwo_score': round(grid.best_score_, 2),
            'random_state': random_state,
            'hyperparameters_searched': hyperparameters
        }
        return description

    def predict(self, data):
        """
        Make predictions based on input data.

        Parameters:
        data (dict): Input data for prediction.

        Returns:
        int: Prediction result.
        """
        # Get model with given name
        # Get training data stats
        # Transform data point with stats (extract features)
        # predict with loaded model
        y_pred = self.model.predict(X=data)
        return y_pred
