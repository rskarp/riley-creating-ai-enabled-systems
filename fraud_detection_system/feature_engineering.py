from datetime import datetime
from typing import Union, List, Dict
import pandas as pd
import numpy as np
import json


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
            dataSource = data
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

    def load(self, output_filename: str, format: str = None) -> None:
        """
        Write self.dataset to file.

        Parameters
        ----------
        output_filename : string
            The path and filename where the dataset will be written to.
        format : string (optional)
            The format of the output file (accept: .csv, .json, .parquet).

        Returns
        -------
        None
        """
        # Get filetype
        fileType = format.split('.')[-1].lower()
        # Create full output filename
        fullFilename = f'{output_filename}_v{self.version}.{fileType}'
        # Save file based on format
        if fileType == 'csv':
            self.dataset.to_csv(fullFilename, index=False)
        elif fileType == 'parquet':
            self.dataset.to_parquet(fullFilename, 'pyarrow')
        elif fileType == 'json':
            self.dataset.to_json(fullFilename, index=False)
        else:  # default to parquet if not specified
            self.dataset.to_parquet(fullFilename, 'pyarrow')

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

    # Sampling

    def _min_max_scale(self, cols: List[str]) -> pd.DataFrame:
        """
        Scale the columns to be between 0 and 1 based on the min and max column values.

        Parameters
        ----------
        cols : List[str]
            The names of columns to scale.

        Returns
        -------
        DataFrame
            The scaled columns of the dataset.
        """
        data = self.dataset.loc[:, cols]
        return (data-data.min())/(data.max()-data.min())

    def _standardize(self, cols: List[str]) -> pd.DataFrame:
        """
        Standardize the columns to have mean 0 and standard deviation 1.

        Parameters
        ----------
        cols : List[str]
            The names of columns to standardize.

        Returns
        -------
        DataFrame
            The standardized columns of the dataset.
        """
        data = self.dataset.loc[:, cols]
        return (data-data.mean())/data.std()

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

    def transform(self, *args, **kwargs) -> pd.DataFrame:
        """
        Transform features in self.dataset.

        Parameters
        ----------
        transformation : str
            The type of transformation to perform: scale, standardize, and add_noise.
        column_names : List[str]
            The list of column_names indicating for which columns to apply the transformation.
        seed : int (optional)
            The random seed value to be used for the random state.

        Returns
        -------
        DataFrame
            The transformed features.
        """
        transformation = kwargs.get('transformation', '')
        cols = kwargs.get('column_names', [])
        seed = kwargs.get('seed')

        if transformation == 'scale':
            return self._min_max_scale(cols)
        elif transformation == 'standardize':
            return self._standardize(cols)
        elif transformation == 'add_noise':
            return self._add_noise(cols, seed)

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

    def describe(self, dataframe: pd.DataFrame, output_file: str = None) -> Dict:
        """
        Generate description of the dataset and measurements assessing the quality/integrity of the features.

        Parameters
        ----------
        dataframe : DataFrame
            The data to describe.
        output_file : str (Optional)
            The path and filename where the description will be written to.

        Returns
        -------
        Dict
            Dictionary containing the description of the dataset and measurements assessing the quality/integrity of the features.
        """
        # Version and data source information is based on self.dataset since that's all that's stored
        # Columns, date ranges, and metrics are based on the dataframe input parameter.
        info = {
            'description': {
                'version': self.version,
                'data sources': self.data_source_names,
                'column names': list(dataframe.columns),
                'date ranges': (dataframe['trans_date_trans_time'].min().strftime('%Y-%m-%d_%H-%M-%S_%f'), dataframe['trans_date_trans_time'].max().strftime('%Y-%m-%d_%H-%M-%S_%f')),
                'num data sources': len(self.data_sources)
            },
            'measures': {
                'numeric_col_correlations': self._get_numeric_col_correlations(dataframe).to_numpy().tolist(),
                'numeric_col_variances': self._get_numeric_col_variances(dataframe).to_dict()
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
        return self.dataset


def test():
    testDir = './data/my_tests/'
    csv = FeatureEngineering('./data/transactions_0.csv')
    print(csv.get_data_source())
    csv.load(f'{testDir}csv_to_json', '.json')
    d = csv.transform(transformation='scale',
                      column_names=['amt', 'unix_time'])
    print(d.describe())
    d = csv.transform(transformation='standardize',
                      column_names=['amt', 'unix_time'])
    print(d.describe())
    d = csv.transform(transformation='add_noise',
                      column_names=['amt', 'unix_time'])
    print(d.describe())
    csv.describe(csv.get_dataset(), f'{testDir}csvLogs')


if __name__ == '__main__':
    test()
