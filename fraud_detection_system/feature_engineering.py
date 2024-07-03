from datetime import datetime
from typing import Union, List, Dict
import pandas as pd
import numpy as np
import os
import json
from imblearn.over_sampling import SMOTENC
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif


class FeatureEngineering():
    """
    This is a class that transforms the sampled dataset, extracts features, and assess the quality of features.

    Attributes:
        dataset : DataFrame
            The transaction data.
        version : str
            The transformations that have been run on the dataset.
        data_sources : List[DataFrame]
            List of DataFrames of source data from whcih self.dataset is constructed.
        data_source_names : List[str]
            List of filenames of source data from whcih self.dataset is constructed.
        transformations : List[str]
            The transformations that have been run on the dataset.
        categorical_feature_scores : Dict
            The chi2 scores for categorical features calculated during categorical feature selection.
        numeric_feature_scores : Dict
            The anova scores for numeric features calculated during numeric feature selection.
        column_stats : Dict
            The means, standard deviations, minimums, and maximums of each column.
    """

    def __init__(self, raw_data):
        """
        Initialize the instance of the FeatureEngineering class.

        Parameters
        ----------
        raw_data : Union[str, pd.DataFrame]
            The Dataframe or filename containing the data.

        Returns
        -------
        None
        """
        self.version = ''
        self.data_sources = []
        self.data_source_names = []
        self.dataset = self.extract_data(raw_data)
        self.labels = self.dataset.loc[:, 'is_fraud']
        # Initialize the list of transformations that have been run on the dataset
        self.transformations = []
        # Initialize dictionary of categorical feature scores
        self.categorical_feature_scores = {}
        # Initialize dictionary of numeric feature scores
        self.numeric_feature_scores = {}
        # Initialize dictionary of column stats
        self.column_stats = {'mean': [], 'std': [], 'min': [], 'max': []}
        self.id_cols = ['Index', 'trans_num']
        self.datetime_cols = ['dob', 'trans_date_trans_time', 'unix_time']
        self.continuous_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'dob',
                                'trans_date_trans_time', 'unix_time', 'year_trans_date_trans_time', 'year_dob']
        self.categorical_cols = ['category', 'cc_num', 'merchant', 'first', 'last', 'sex', 'street', 'city', 'zip', 'job', 'month_trans_date_trans_time',
                                 'state', 'day_of_week_trans_date_trans_time', 'hour_of_day_trans_date_trans_time', 'month_dob', 'day_of_week_dob', 'hour_of_day_dob']

    # Deployment

    def generate_features(self, version: str, stats: Dict = None, run_smote: bool = False, random_state: int = 1) -> Dict:
        '''
        Generate features from the given dataset version.

        Parameters:
        version : str
            The version name to use for the data source.
        stats : Dict
            Dictionary of column stats used for scale and standardize steps.
        run_smote : bool (Optional. Default False)
            Boolean flag indicating whether or not to run SMOTE if is_fraud classes are imbalanced.
        random_state : int (Optional. Default 1)
            The random_state value to use for reproducibility.

        Returns:
        Dict:
            Description of the features.
        '''
        # Apply transformations to feature data
        # Remove identifier columns
        self.transform(
            transformation='remove', column_names=self.id_cols, inplace=True)
        # Convert dates to numbers
        self.transform(
            transformation='to_numeric', column_names=self.datetime_cols, inplace=True)
        # Convert categorical columns to category type
        self.transform(
            transformation='to_category', column_names=self.categorical_cols, inplace=True)

        if run_smote == True:
            fraud_proportion = (self.get_dataset()["is_fraud"] == 1).sum(
            )/self.get_dataset().shape[0]
            # If fraud proportion is unbalanced,
            #   oversample using SMOTE to increase the number or fraudlent training samples
            if abs(0.5-fraud_proportion) > 0.1:
                self.transform(
                    transformation='oversample', seed=random_state, inplace=True)

        # Additional transformations after SMOTE
        col_stats = stats if stats is not None else self.calculate_column_stats()
        # Encode categorical columns
        self.transform(
            transformation='encode', column_names=self.categorical_cols, inplace=True)
        # Standardize
        self.transform(transformation='standardize',
                       stats=col_stats, column_names=self.continuous_cols, inplace=True)
        # Scale
        self.transform(
            transformation='scale', stats=col_stats, column_names=self.continuous_cols, inplace=True)
        # Add Gaussian noise
        self.transform(
            transformation='add_noise', column_names=self.continuous_cols, inplace=True, seed=random_state)

        # Find the best features from the training data
        percentile = 50
        categorical_data = self.transform(
            transformation='select_categorical_features', percentile=percentile, column_names=self.categorical_cols)
        numeric_data = self.transform(
            transformation='select_numeric_features', percentile=percentile, column_names=self.continuous_cols)
        # Filter features to only include the best features
        self.transform(transformation='filter_columns',
                       column_names=[*categorical_data.columns,
                                     *numeric_data.columns],
                       inplace=True)

        # Add labels back in after transformations
        self.dataset['is_fraud'] = self.labels
        # Save features to file
        format = 'parquet'
        os.makedirs('resources/features', exist_ok=True)
        file_path = f"resources/features/{version}"
        self.load(file_path, format, False)
        # Return description
        return self.describe(self.dataset, col_stats=col_stats, version=version)

    def generate_test_features(self, stats: Dict = None, random_state: int = 1, filter_cols: List[str] = []) -> pd.DataFrame:
        '''
        Generate features for test/production data.

        Parameters:
        stats : Dict
            Dictionary of column stats used for scale and standardize steps.
        random_state : int (Optional. Default 1)
            The random_state value to use for reproducibility.
        filter_cols : List[str]
            The names of columns to include in the feature data.

        Returns:
        Dataframe:
            The generated features.
        '''
        # Remove identifier columns
        self.transform(
            transformation='remove', column_names=self.id_cols, inplace=True)
        # Convert dates to numbers
        self.transform(
            transformation='to_numeric', column_names=self.datetime_cols, inplace=True)
        # Convert categorical columns to category type
        self.transform(
            transformation='to_category', column_names=self.categorical_cols, inplace=True)
        # Encode categorical columns
        self.transform(
            transformation='encode', column_names=self.categorical_cols, inplace=True)
        # Standardize
        self.transform(transformation='standardize',
                       stats=stats, column_names=self.continuous_cols, inplace=True)
        # Scale
        self.transform(
            transformation='scale', stats=stats, column_names=self.continuous_cols, inplace=True)
        # Add Gaussian noise
        self.transform(
            transformation='add_noise', column_names=self.continuous_cols, inplace=True, seed=random_state)
        # Filter features to only include the best features
        self.transform(transformation='filter_columns',
                       column_names=filter_cols,
                       inplace=True)
        # Add labels back in after transformations
        self.dataset['is_fraud'] = self.labels
        return self.get_dataset()

    # Extract and Load

    def extract_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Read data from either a file (.csv, .json, or .parquet) or a pandas.DataFrame.

        Parameters
        ----------
        data : Union[str, pd.DataFrame]
            The DataFrame or filename containing the data to be read.

        Returns
        -------
        DataFrame
            The updated dataset containing the newly read data.
        """
        # Get new data as DataFrame
        if isinstance(data, pd.DataFrame):
            dataSource = data.copy(deep=True)
            dataSourceName = 'DataFrame'
        else:
            dataSourceName = data
            fileType = data.split('.')[-1].lower()
            if fileType == 'csv':
                dataSource = pd.read_csv(data)
            elif fileType == 'parquet':
                dataSource = pd.read_parquet(data, 'pyarrow')
            elif fileType == 'json':
                dataSource = pd.read_json(data)
            else:
                raise ValueError(
                    'File must be of type .csv, .json, or .parquet')

        # Store the new data source and name
        self.data_sources.append(dataSource)
        self.data_source_names.append(dataSourceName)

        # Concatenate new DataFrame with existing dataset (this assumes columns are the same)
        if hasattr(FeatureEngineering, 'dataset'):
            self.dataset = pd.concat(
                [self.dataset, dataSource], ignore_index=True)
        else:
            self.dataset = dataSource

        # Set version
        self.set_version()

        return self.dataset

    def load(self, output_filename: str, format: str = None, add_version: bool = True) -> None:
        """
        Write self.dataset to file.

        Parameters
        ----------
        output_filename : string
            The path and filename where the dataset will be written to.
        format : string (optional)
            The format of the output file (accept: .csv, .json, .parquet).
        add_version : bool (Optional. Default True)
            Boolean flag indicating whether to append self.version to filename.

        Returns
        -------
        str:
            The full filename of the saved file.
        """
        # Get filetype
        fileType = format.split('.')[-1].lower()
        # Create full output filename
        version = f'_v{self.version}' if add_version else ''
        fullFilename = f'{output_filename}{version}.{fileType}'
        # Save file based on format
        if fileType == 'csv':
            self.dataset.to_csv(fullFilename, index=False)
        elif fileType == 'parquet':
            self.dataset.to_parquet(fullFilename, engine='pyarrow')
        elif fileType == 'json':
            self.dataset.to_json(fullFilename, index=False)
        else:  # default to parquet if not specified
            self.dataset.to_parquet(fullFilename, engine='pyarrow')
        return fullFilename

    # Description

    def get_data_source(self) -> List[str]:
        """
        Indicates from which file(s) the dataset is constructed.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            The list of filenames from which the dataset is constructed.
        """
        return self.data_source_names

    def _get_current_datetime_str(self) -> str:
        """
        Get the current datetime as a formatted string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Current datetime in the format '%Y-%m-%d_%H-%M-%S_%f'.
        """
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')

    def set_version(self) -> str:
        """
        Gives the dataset a unique identifier.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The unique identifier for the dataset.
        """
        # Use current datetime as unique identifier
        ver = self._get_current_datetime_str()
        self.version = ver
        return ver

    # Transformations

    def calculate_column_stats(self, dataframe: pd.DataFrame = None) -> Dict:
        """
        Calculate the column statistics for self.dataset.

        Parameters
        ----------
        dataframe : DataFrame (Optional)
            The dataframe to calculate statistics for. If empty, self.dataset will be used.

        Returns
        -------
        Dict
            The means, standard deviations, minimums, maximums, variances, and correlations of the dataframe.
        """
        numeric_data = dataframe.select_dtypes(
            'number') if dataframe is not None else self.dataset.select_dtypes('number')
        self.column_stats = {
            'mean': numeric_data.mean(),
            'std': numeric_data.std(),
            'min': numeric_data.min(),
            'max': numeric_data.max(),
            'var': numeric_data.var(),
            'corr': numeric_data.corr(),
        }
        return self.column_stats

    def _min_max_scale(self, cols: List[str], stats: Dict = None) -> pd.DataFrame:
        """
        Scale the columns to be between 0 and 1 based on the min and max column values.

        Parameters
        ----------
        cols : List[str]
            The names of columns to scale.
        stats : Dict (optional)
            The means, standard deviations, minimums, and maximums of each column.

        Returns
        -------
        DataFrame
            The scaled columns of the dataset.
        """
        # Calculate stats if not provided
        colStats = self.calculate_column_stats() if stats is None else stats
        # Get scaling stats
        mins = colStats['min'][cols]
        maxs = colStats['max'][cols]
        # Replace 0 with 1 to avoid division by 0
        diffs = (maxs-mins).replace(0, 1)

        data = self.dataset.loc[:, cols]
        return (data-mins)/diffs

    def _standardize(self, cols: List[str], stats: Dict = None) -> pd.DataFrame:
        """
        Standardize the columns to have mean 0 and standard deviation 1.

        Parameters
        ----------
        cols : List[str]
            The names of columns to standardize.
        stats : Dict (optional)
            The means, standard deviations, minimums, and maximums of each column.

        Returns
        -------
        DataFrame
            The standardized columns of the dataset.
        """
        # Calculate stats if not provided
        colStats = self.calculate_column_stats() if stats is None else stats
        # Get standardization stats
        means = colStats['mean'][cols]
        stds = colStats['std'][cols]
        # Replace 0 with 1 to avoid division by 0
        stds.replace(0, 1, inplace=True)
        data = self.dataset.loc[:, cols]
        return (data-means)/stds

    def _add_noise(self, cols: List[str], seed: int = None) -> pd.DataFrame:
        """
        Add Gaussian random noise to the features.

        Parameters
        ----------
        cols : List[str]
            The names of columns to add noise to.
        seed : int (optional)
            The random seed value to be used for the random state.

        Returns
        -------
        DataFrame
            The noisy columns of the dataset.
        """
        data = self.dataset.loc[:, cols]
        if seed != None:
            np.random.seed(seed)
        noise = np.random.normal(0, 1, data.shape)
        return data + noise

    def _encode(self, cols: List[str]) -> pd.DataFrame:
        """
        Encode the columns of categorical values.

        Parameters
        ----------
        cols : List[str]
            The names of the columns to encode.

        Returns
        -------
        DataFrame
            The integer encoded columns of the dataset.
        """
        data = self.dataset.loc[:, cols]
        for c in data.columns:
            if data.dtypes[c].name == 'category':
                data[c] = data[c].cat.codes
        return data

    def _to_category(self, cols: List[str]) -> pd.DataFrame:
        """
        Encode the column of categorical values.

        Parameters
        ----------
        col : str
            The name of the column to encode.

        Returns
        -------
        DataFrame
            The integer encoded column of the dataset.
        """
        data = self.dataset.loc[:, cols]
        for c in data.columns:
            if data.dtypes[c].name == 'string':
                data[c] = data[c].str.lower().astype('category')
            elif data.dtypes[c].name in ['int32', 'int64']:
                data[c] = data[c].astype('category')
        return data

    def _oversample(self, seed: int = None) -> pd.DataFrame:
        """
        Oversample the minority class.

        Parameters
        ----------
        seed : int (optional)
            The random seed value to be used for the random state.

        Returns
        -------
        DataFrame
            The datset with the syntehtic samples generated by oversampling the minorty class.
        """
        X = self.dataset.loc[:, self.dataset.columns != 'is_fraud']
        y = self.dataset.loc[:, 'is_fraud']
        sm = SMOTENC(random_state=seed, categorical_features='auto')
        X_res, y_res = sm.fit_resample(X, y)
        X_res['is_fraud'] = y_res
        return X_res

    def _select_categorical_features(self, cols: List[str], percentile: int) -> pd.DataFrame:
        """
        Select the best categorical features using Chi-Square test.

        Parameters
        ----------
        cols : List[str]
            The names of the categorical columns to choose from.
        percentile : int
            The percentile indicating how many of the top features to include.

        Returns
        -------
        DataFrame
            The datset with only the best categorical features.
        """
        # Make sure is_fraud isn't in feature data
        cleanCols = [c for c in cols if c != 'is_fraud']
        # Create features and target
        X = self.dataset.loc[:, cleanCols]
        y = self.dataset.loc[:, 'is_fraud']
        # Select top percentile features with highest chi-squared statistics
        chi2_selector = SelectPercentile(chi2, percentile=percentile)
        chi2_selector.fit(X, y)

        # Save scores returned from the selector for each feature
        self.categorical_feature_scores = dict(
            zip(X.columns, chi2_selector.scores_))

        # Return the best features
        bestCols = np.asarray(X.columns)[chi2_selector.get_support()]
        return self.dataset.loc[:, bestCols]

    def _select_numeric_features(self, cols: List[str], percentile: int) -> pd.DataFrame:
        """
        Select the best numeric features using ANOVA test.

        Parameters
        ----------
        cols : List[str]
            The names of the numeric columns to choose from.
        percentile : int
            The percentile indicating how many of the top features to include.

        Returns
        -------
        DataFrame
            The datset with only the best numeric features.
        """
        # Make sure is_fraud isn't in feature data
        cleanCols = [c for c in cols if c != 'is_fraud']
        # Create features and target
        X = self.dataset.loc[:, cleanCols]
        y = self.dataset.loc[:, 'is_fraud']
        # Select top percentile features with highest anova statistics
        anova_selector = SelectPercentile(f_classif, percentile=percentile)
        anova_selector.fit(X, y)

        # Save scores returned from the selector for each feature
        self.numeric_feature_scores = dict(
            zip(X.columns, anova_selector.scores_))

        # Return the best features
        bestCols = np.asarray(X.columns)[anova_selector.get_support()]
        return self.dataset.loc[:, bestCols]

    def transform(self, *args, **kwargs) -> pd.DataFrame:
        """
        Transform features in self.dataset.

        Parameters
        ----------
        transformation : str
            The type of transformation to perform: scale, standardize, add_noise, encode, remove, oversample, to_category, to_numeric, filter_columns, select_categorical_features, or select_numeric_features.
        column_names : List[str]
            The list of column_names indicating for which columns to apply the transformation.
        inplace : bool (deafult = False)
            Boolean indicating whether to apply the transformation to self.dataset, or just return the transformed columns.
        seed : int (optional)
            The random seed value to be used for the random state.

        Returns
        -------
        DataFrame
            The dataframe with transformed features.
        """
        seed = kwargs.get('seed')
        transformation = kwargs.get('transformation')
        column_names = kwargs.get('column_names')
        inplace = kwargs.get('inplace')
        percentile = kwargs.get('percentile')
        stats = kwargs.get('stats')

        transformed_dataframe = self.dataset.copy()

        if transformation == 'scale':
            transformed_dataframe[column_names] = self._min_max_scale(
                column_names, stats)
        elif transformation == 'standardize':
            transformed_dataframe[column_names] = self._standardize(
                column_names, stats)
        elif transformation == 'add_noise':
            transformed_dataframe.loc[:, column_names] = self._add_noise(
                column_names, seed)
        elif transformation == 'encode':
            transformed_dataframe[column_names] = self._encode(
                column_names)
        elif transformation == 'remove':
            transformed_dataframe.drop(columns=column_names, inplace=True)
        elif transformation == 'to_numeric':
            transformed_dataframe[column_names] = transformed_dataframe[column_names].astype(
                'int64')
        elif transformation == 'to_category':
            transformed_dataframe[column_names] = self._to_category(
                column_names)
        elif transformation == 'oversample':
            transformed_dataframe = self._oversample(seed)
        elif transformation == 'select_categorical_features':
            transformed_dataframe = self._select_categorical_features(
                column_names, percentile)
        elif transformation == 'select_numeric_features':
            transformed_dataframe = self._select_numeric_features(
                column_names, percentile)
        elif transformation == 'filter_columns':
            transformed_dataframe = transformed_dataframe[column_names]
        else:
            return

        if inplace == True:
            self.dataset = transformed_dataframe.copy()
            self.transformations.append(transformation)

        return transformed_dataframe

    # Measures and Logging

    def _get_numeric_col_correlations(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlations of numeric columns.

        Parameters
        ----------
        dataframe : DataFrame
            The data to retrieve the metrics from.

        Returns
        -------
        DataFrame
            The variances for each pair of numeric columns.
        """
        return dataframe.select_dtypes('number').corr()

    def _get_numeric_col_variances(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate the variances of numeric columns.

        Parameters
        ----------
        dataframe : DataFrame
            The data to retrieve the metrics from.

        Returns
        -------
        Series
            The variances for each numeric column.
        """
        return dataframe.select_dtypes('number').var()

    def describe(self, dataframe: pd.DataFrame, col_stats: Dict = None, output_file: str = None, version: str = None) -> Dict:
        """
        Generate description of the dataset and measurements assessing the quality/integrity of the features.

        Parameters
        ----------
        dataframe : DataFrame
            The data to describe.
        col_stats : Dict
            The column statistics to store.
        output_file : str (Optional)
            The path and filename where the description will be written to.
        version : str (Optional. Default None)
            The version name to use instead of self.version.
        Returns
        -------
        Dict
            Dictionary containing the description of the dataset and measurements assessing the quality/integrity of the features.
        """
        # Version and data source information is based on self.dataset since that's all that's stored
        # Columns, date ranges, and metrics are based on the dataframe input parameter.
        if 'trans_date_trans_time' in dataframe.columns:
            date_ranges = (pd.to_datetime(dataframe['trans_date_trans_time'].min()).strftime('%Y-%m-%d_%H-%M-%S_%f'),
                           pd.to_datetime(dataframe['trans_date_trans_time'].max()).strftime('%Y-%m-%d_%H-%M-%S_%f'))
        else:
            date_ranges = []

        stats = col_stats if col_stats is not None else self.calculate_column_stats(
            dataframe)
        info = {
            'version': version if version is not None else self.version,
            'data_sources': self.data_source_names,
            'column_names': list(dataframe.columns),
            'date_ranges': date_ranges,
            'num_data_sources': len(self.data_sources),
            'transformations': self.transformations,
            'categorical_feature_chi2_scores': self.categorical_feature_scores,
            'numeric_feature_anova_scores': self.numeric_feature_scores,
            'measures': {
                'numeric_col_correlations': stats['corr'].to_numpy().tolist(),
                'numeric_col_variances': stats['var'].to_dict(),
                'numeric_col_means': stats['mean'].to_dict(),
                'numeric_col_standard_deviations': stats['std'].to_dict(),
                'numeric_col_minimums': stats['min'].to_dict(),
                'numeric_col_maximums': stats['max'].to_dict()
            }
        }

        # Save description to .json file if output_file is given
        if output_file != None:
            curDatetime = self._get_current_datetime_str()
            baseName = output_file.replace('.json', '')
            fullFilename = f'{baseName}_v{self.version}_{curDatetime}.json'
            with open(fullFilename, 'w') as fp:
                json.dump(info, fp, indent=4)

        # Return info
        return info

    def get_dataset(self) -> pd.DataFrame:
        """
        Return the dataset in its current state.

        Parameters
        ----------
        None

        Returns
        -------
        DataFrame
            Contains all data as it currently exists in the dataset. 
        """
        return self.dataset.copy(deep=True)


def test():
    csv = FeatureEngineering('./data/transactions_0.csv')
    print(csv.get_data_source())
    csv.load(f'./data/logs/csv_to_json', '.json')
    d = csv.transform(transformation='scale',
                      column_names=['amt', 'unix_time'])
    print(d.describe())
    d = csv.transform(transformation='standardize',
                      column_names=['amt', 'unix_time'])
    print(d.describe())
    d = csv.transform(transformation='add_noise',
                      column_names=['amt', 'unix_time'])
    print(d.describe())
    csv.describe(csv.get_dataset(), f'./data/logs/csvLogs')


if __name__ == '__main__':
    test()
