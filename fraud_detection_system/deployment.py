import datetime
import json
import os
from typing import Dict, List, Union
import pandas as pd
from dataset import Dataset
from data_engineering import DataEngineering
from feature_engineering import FeatureEngineering
from model import Model
from metrics import Metrics
from enum import Enum

Endpoint = Enum('Endpoint', ['generate_new_dataset', 'generate_new_features', 'dataset_description', 'dataset_features_description',
                             'model_description', 'dataset_list', 'feature_list', 'model_list', 'model_metrics',
                             'predict', 'train', 'system_metrics'])


class DeploymentPipeline:
    """
    DeploymentPipeline class handles the deployment logic for ML models,
    including training and inference pipelines.
    """

    def __init__(self):
        """
        Initialize the DeploymentPipeline with necessary components.
        """
        print('Initializing deployment pipline. Please wait...')
        self.dataset_train = self.get_train_source()
        self.dataset_test = self.get_test_source()
        self.metrics = None
        self.model = Model()
        self.endpoint_usage = {}
        self.allCols = [
            "Index",
            "trans_date_trans_time",
            "cc_num",
            "merchant",
            "category",
            "amt",
            "first",
            "last",
            "sex",
            "street",
            "city",
            "state",
            "zip",
            "lat",
            "long",
            "city_pop",
            "job",
            "dob",
            "trans_num",
            "unix_time",
            "merch_lat",
            "merch_long",
            "is_fraud"
        ]
        print('Pipeline initilaized')

    def track_endpoint_usage(self, endpoint: Endpoint):
        '''
        Keep track of how many times each endpoint is called for system monitoring.

        Parameters:
        endpoint (Endpoint) : the endpoint that was called.

        Returns:
        None
        '''
        self.endpoint_usage[endpoint.name] = self.endpoint_usage.get(
            endpoint.name, 0) + 1

    def get_system_metrics(self) -> Dict:
        '''
        Get the metrics of endpoint usage of the system.

        Parameters:
        None

        Returns:
        Dict : the metrics of how many times each endpoint has been called since system start up.
        '''
        return self.endpoint_usage

    def clean_raw_data(self, data: Union[str, pd.DataFrame], predict: bool = False) -> Dataset:
        '''
        Clean the raw data.

        Parameters:
        data : data: Union[str, pd.DataFrame]
            Name of the file or the dataframe containing the raw data to be extracted.
        predict : boolean
            Boolean indicating whether the data is to be predicted (therefore is unlabeled)

        Returns:
        Dataframe:
            The cleaned data.
        '''
        # Perform all cleaning steps
        data_eng = DataEngineering(data)
        data_eng.clean_missing_values()
        data_eng.remove_duplicates()
        data_eng.standardize_dates('trans_date_trans_time')
        data_eng.standardize_dates('unix_time')
        data_eng.standardize_dates('dob')
        data_eng.resolve_anomalous_dates('trans_date_trans_time')
        data_eng.resolve_anomalous_dates('unix_time')
        data_eng.resolve_anomalous_dates('dob')
        data_eng.expand_dates('trans_date_trans_time')
        data_eng.expand_dates('dob')
        data_eng.trim_spaces('merchant')
        data_eng.trim_spaces('city')
        data_eng.trim_spaces('first')
        data_eng.trim_spaces('last')
        data_eng.trim_spaces('street')
        data_eng.trim_spaces('job')

        # Return Dataset with cleaned and labeled data
        cleaned = Dataset(data_eng.get_dataset())
        if predict == False:
            cleaned.sample(sampling_method='labeled', inplace=True)
        return cleaned

    def get_train_source(self) -> Dataset:
        '''
        Get dataset containing training data (from the first 2 raw data files).

        Parameters:
        None

        Returns:
        Dataset:
            The cleaned and labeled training data.
        '''
        # Read first 2 raw data files into one dataset
        raw_dataset = Dataset(raw_data='./data/transactions_0.csv')
        raw_dataset.extract_data('./data/transactions_1.parquet')
        # Save all raw data in one parquet file
        filename = raw_dataset.load(
            './data/transactions_1-2', '.parquet', False)
        # Clean data
        return self.clean_raw_data(filename)

    def get_test_source(self) -> Dataset:
        '''
        Get dataset containing testing data (from the 3rd raw dataset)

        Parameters:
        None

        Returns:
        Dataset:
            The cleaned and labeled testing data.
        '''
        # Use 3rd raw dataset as test data
        return self.clean_raw_data('./data/transactions_2.json')

    def _prepare_dataset_for_model_inference(self, model_version: str, data: pd.DataFrame, random_state: int = 1, predict: bool = False, model_info: Dict = None) -> pd.DataFrame:
        '''
        Extract features from given data using transformation steps used to train the given model version.

        Parameters:
        model_version (str): The model version to use to determine transformation parameters (for scale, standardize, and best feature filter).
        data (DataFrame): The raw data to be cleaned, transformed, and features extracted.
        random_state (int, Optional. Default = 1): The random_state value to use for reproducibility.
        predict (bool, Optional. Default = False): Boolean indicating whether the data is to be predicted (therefore is unlabeled)
        model_info (Dict, Optional. Default = None): Model information used to find stats used for training the model.

        Returns:
        DataFrame: the features extracted from the dataset.
        '''
        # Get name of training dataset from model log
        model_log = model_info if model_info is not None else self.get_log(
            'models', model_version)
        features_log = self.get_log(
            'features', model_log['description']['train_dataset'])
        # Get training data column stats from dataset features log, so we can use them for test data transformation
        stats = features_log['description']['measures']
        col_stats = {
            'mean': pd.Series(stats['numeric_col_means']),
            'std': pd.Series(stats['numeric_col_standard_deviations']),
            'min': pd.Series(stats['numeric_col_minimums']),
            'max': pd.Series(stats['numeric_col_maximums'])
        }
        # Get column names used in training
        colNames = features_log['description']['column_names']
        # Clean input data
        features = FeatureEngineering(
            self.clean_raw_data(data, predict).get_dataset())
        # Transform and extract features from input data
        return features.generate_test_features(
            stats=col_stats, random_state=random_state, filter_cols=colNames)

    def predict(self, model_version: str, data: Dict, random_state: int = 1):
        """
        Make predictions using the deployed model.

        Parameters:
        model_version (str): name of the model version to use for inference.
        data (dict): Input data for prediction.
        random_state (int): The random_state value to use for reproducibility.

        Returns:
        ndarray: Prediction results.
        """
        # Load the specified model version
        try:
            self.model.set_model(f'resources/models/{model_version}.joblib')
        except FileNotFoundError:
            raise FileNotFoundError(f'Model not {model_version} found')
        # Get features for data point
        df = pd.DataFrame(data, index=[0], columns=self.allCols)
        features = self._prepare_dataset_for_model_inference(
            model_version, df, random_state, True)
        # Remove label column for prediction
        features = features.loc[:, features.columns != 'is_fraud']
        if features.shape[0] < 1:
            raise ValueError(
                'Data must have all required features (cc_num, amt, trans_num).')
        # Predict using the model
        prediction = self.model.predict(features)
        return prediction

    def get_log(self, component, reference):
        """
        Get the description of a component (e.g., dataset, model, etc.) from the logs.

        Parameters:
        component (str): The component name.
        reference (str): The reference ID for the logs.

        Returns:
        dict: Description from the logs.
        """
        log_path = f"resources/logs/{component}/{reference}.json"
        try:
            with open(log_path, "r") as logs:
                description = json.load(logs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found: {log_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from log file: {log_path}")
        return description

    def generate_new_dataset(self, dataset_version: str, dataset_type: str, n_samples: int = 1000, sampling_type: str = 'stratified', random_state: int = 1) -> Dict:
        '''
        Generate a new dataset sampled from the training set.

        Parameters:
        dataset_version : str
            The version name to save the data to.
        dataset_type : str
            The type of datset to create: train or test.
        n_samples : int (Optional. Default 1000)
            The number of samples (if random) or samples per class (if stratified) to be sampled.
        sampling_type : str (Optional. Default 'stratified')
            The sampling method to use: random or stratified.
        random_state : int (Optional. Default 1)
            The random_state value to use for reproducibility.

        Returns:
        Dict:
            Description of the dataset.
        '''
        if dataset_type == 'train':
            return self.dataset_train.generate_dataset(dataset_version, n_samples, sampling_type, random_state)
        elif dataset_type == 'test':
            return self.dataset_test.generate_dataset(dataset_version, n_samples, sampling_type, random_state)

    def generate_new_features(self, dataset_version: str, run_smote: bool = False, random_state: int = 1) -> Dict:
        '''
        Generate a features from the given dataset version.

        Parameters:
        dataset_version : str
            The version name to use for the data source.
        run_smote : bool (Optional. Default False)
            Boolean flag indicating whether or not to run SMOTE if is_fraud classes are imbalanced.
        random_state : int (Optional. Default 1)
            The random_state value to use for reproducibility.

        Returns:
        Dict:
            Description of the features.
        '''
        features = FeatureEngineering(
            f'resources/datasets/{dataset_version}.parquet')
        return features.generate_features(dataset_version, run_smote=run_smote, random_state=random_state)

    def train(self, model_version: str, model_type: str, dataset_version: str, hyperparameters: Dict, random_state: int = 1) -> Dict:
        '''
        Trains the specified type of model on the specified data.

        Parameters:
        model_version : str
            The name of the model version for this model.
        model_type : str
            The type of model to train: random_forest, stocahstic_gradient_descent, or logistic_regression.
        dataset_version : str
            The dataset to use for training data.
        hyperparameters : Dict
            The hyperparameters used for tuning. The value of each key in the Dict should be a list.
        random_state : int (Optoinal. Default 1)
            The random_state value to use for reproducibility.

        Returns:
        Dict : 
            Description from the logs.
        '''
        description = self.model.train(
            model_version, model_type, dataset_version, hyperparameters, random_state)
        data = pd.read_parquet(
            f'resources/datasets/{dataset_version}.parquet', engine='pyarrow')
        features = self._prepare_dataset_for_model_inference(
            model_version, data, random_state, model_info={'description': description})
        metrics = Metrics(features, model_version)
        description['train_metrics'] = metrics.run_metrics()
        return description

    def get_model_metrics(self, model_version: str, dataset_version: str, random_state: int = 1) -> Dict:
        """
        Get the model metrics for a given model and dataset version.

        Parameters:
        dataset_version (str): The version of the dataset to use for inference.
        model_version (str): The version of the model to use for predictions.
        random_state (int, Optional. Default 1):  The random_state value to use for reproducibility.

        Returns:
        Dict: Dictionary containing the model metrics.
        """
        data = pd.read_parquet(
            f'resources/datasets/{dataset_version}.parquet', engine='pyarrow')
        features = self._prepare_dataset_for_model_inference(
            model_version, data, random_state)
        metrics = Metrics(features, model_version)
        return metrics.run_metrics()

    def get_resource_list(self, resourceType: str) -> List[str]:
        '''
        Returns list of available resources of the given type.

        Parameters:
        resourceType : str
            The type of resources to list (dataset, feature, or model)

        Returns:
        List[str] : 
            List of versions available.
        '''
        if resourceType == 'datasets':
            path = 'resources/datasets'
        elif resourceType == 'models':
            path = 'resources/models'
        elif resourceType == 'features':
            path = 'resources/features'
        return [''.join(f.split('.')[:-1]) for f in os.listdir(path)]

    def dataset_has_features(self, dataset_version: str) -> bool:
        '''
        Returns boolean indicating whether features have been extracted for the dataset.

        Parameters:
        dataset_version : str
            The dataset version name.

        Returns:
        Bool : 
            Boolean indicating whether features have been extracted for the dataset.
        '''
        return dataset_version in [''.join(f.split('.')[:-1]) for f in os.listdir('resources/features')]

    def log(self, component, log_entry, log_file):
        """
        Log the prediction details to a specified log file.
        """
        os.makedirs(f'resources/logs/{component}', exist_ok=True)
        with open(f'resources/logs/{component}/{log_file}', "w") as log:
            json.dump(log_entry, log, indent=4)
