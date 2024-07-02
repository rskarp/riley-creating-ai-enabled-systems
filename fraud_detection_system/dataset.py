from datetime import datetime
from typing import Union, List, Dict, Tuple
import pandas as pd
import json

from data_engineering import DataEngineering

import os
from sklearn.datasets import make_classification


class Dataset():
    """
    This is a class for constructing a dataset from raw credit card transaction data.

    Attributes:
        dataset : DataFrame
            The transaction data.
        version : str
            The unique version identifier the dataset.
        data_sources : List[DataFrame]
            List of DataFrames of source data from whcih self.dataset is constructed.
        data_source_names : List[str]
            List of filenames of source data from whcih self.dataset is constructed.
        transformations : List[str]
            The transformations that have been run on the dataset.
        format : str
            The file format to use when saving data.
    """

    def __init__(self, raw_data: Union[str, pd.DataFrame] = None, format: str = 'parquet'):
        """
        Initialize the instance of the Dataset class.

        Parameters
        ----------
        raw_data : Union[str, pd.DataFrame]
            The Dataframe or filename containing the data.
        format : str
            The file format to use when saving data.

        Returns
        -------
        None
        """
        self.version = ''
        self.format = format
        self.data_sources = []
        self.data_source_names = []
        self.dataset = pd.DataFrame() if raw_data is None else self.extract_data(raw_data)
        self.transformations = []

    # Deployment

    def generate_dataset(self, version, n_samples: int = 1000, sampling_type: str = 'stratified', random_state: int = 1) -> Dict:
        '''
        Generate a new dataset sampled from self.dataset.

        Parameters:
        version: str
            The version name to save the data to.
        n_samples: int (Optional. Default 1000)
            The number of samples (if random) or sampels per class (if stratified) to be sampled.
        sampling_type: str (Optional. Default 'stratified')
            The sampling method to use: random or stratified.
        random_state: int (Optional. Default 1)
            The random_state value to use for reproducibility.

        Returns:
        Dict:
            Description of the dataset.
        '''
        sampled_data = self.sample(sampling_method=sampling_type, N=n_samples, stratification_factors=[
                                   'is_fraud'], seed=random_state)

        self.deployment_load(version, sampled_data)
        description = self.describe(dataframe=sampled_data, version=version)
        description['random_state'] = random_state
        description['n_classes'] = 2
        return description

    def deployment_load(self, dataset_version: str, dataset_df: pd.DataFrame):
        """
        Load the dataset based on the specified version and file format.

        Parameters:
        dataset_version : str
            The version of dataset to load.
        dataset_df : DataFrame
            The data to load to the given version.

        Returns:
        pd.DataFrame:
            The loaded dataset.
        """

        os.makedirs('resources/datasets', exist_ok=True)
        file_path = f"resources/datasets/{dataset_version}.{self.format}"

        if self.format == 'csv':
            return dataset_df.to_csv(file_path, index=False)
        elif self.format == 'parquet':
            return dataset_df.to_parquet(file_path, engine='pyarrow')
        elif self.format == 'json':
            return dataset_df.to_json(file_path, orient='records', lines=True)

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
        if hasattr(Dataset, 'dataset'):
            self.dataset = pd.concat(
                [self.dataset, dataSource], ignore_index=True)
        else:
            self.dataset = dataSource

        # Set version
        self.set_version()

        return self.dataset

    def load(self, output_filename: str, format: str = None, add_version: bool = True) -> str:
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

    # Sampling

    def _random_sample(self, N: int, seed: int = None) -> pd.DataFrame:
        """
        Randomly select samples from self.dataset.

        Parameters
        ----------
        N : int
            The number of rows to return.
        seed : int (optional)
            The random seed value to be used for the random state.

        Returns
        -------
        DataFrame
            The samples selected as a subset from self.dataset.
        """
        return self.dataset.sample(n=N, random_state=seed)

    def _stratified_sample(self, N: int, strat_factors: List[str], seed: int = None) -> pd.DataFrame:
        """
        Select samples from self.dataset.

        Parameters
        ----------
        N : int
            The number of rows to return per stratum if using stratified sampling.
        stratification_factors : List[str]
            The names of columns to be used as stratification factors.
        seed : int (optional)
            The random seed value to be used for the random state.

        Returns
        -------
        DataFrame
            The samples selected as a subset from self.dataset.
        """
        return self.dataset.groupby([f.lower() for f in strat_factors], group_keys=False, observed=False).apply(lambda x: x.sample(min(len(x), N), random_state=seed))

    def sample(self, *args, **kwargs) -> pd.DataFrame:
        """
        Select samples from self.dataset.

        Parameters
        ----------
        N : int
            The number of rows to return per stratum if using stratified sampling, or total for random sampling.
        sampling_method : str
            The sampling method to use: random or stratified or labeled.
        stratification_factors : List[str]
            The names of columns to be used as stratification factors.
        inplace : bool (deafult = False)
            Boolean indicating whether to apply the sampling to self.dataset, or just return the sampled data.
        seed : int (optional)
            The random seed value to be used for the random state.

        Returns
        -------
        DataFrame
            The samples selected as a subset from self.dataset.
        """
        sampMethod = kwargs.get('sampling_method', '')
        N = kwargs.get('N', 0)
        seed = kwargs.get('seed')
        inplace = kwargs.get('inplace')
        strat_factors = kwargs.get('stratification_factors', [])

        sampled_dataset = self.dataset.copy()
        if sampMethod == 'random':
            sampled_dataset = self._random_sample(N, seed)
        elif sampMethod == 'stratified':
            sampled_dataset = self._stratified_sample(N, strat_factors, seed)
        elif sampMethod == 'labeled':
            sampled_dataset = self.dataset.loc[self.dataset['is_fraud'] >= 0, :]

        if inplace == True:
            self.dataset = sampled_dataset
            self.transformations.append(sampMethod)

        return sampled_dataset

    # Measures and Logging

    def _get_mean_time_since_transaction(self, dataframe: pd.DataFrame) -> float:
        """
        Calculate the mean time since transaction occurrence in days.

        Parameters
        ----------
        dataframe : DataFrame
            The data to retrieve the metric from.

        Returns
        -------
        float
            The mean time since transaction in days.
        """
        return abs(datetime.now()-dataframe['trans_date_trans_time']).mean().days

    def _get_table_completness(self, dataframe: pd.DataFrame) -> float:
        """
        Calculate the table completeness metric.

        Parameters
        ----------
        dataframe : DataFrame
            The data to retrieve the metric from.

        Returns
        -------
        float
            The table completeness.
        """
        return (~dataframe.isna()).sum().sum()/dataframe.size

    def _get_mean_col_completness(self, dataframe: pd.DataFrame) -> float:
        """
        Calculate the mean column completeness metric.

        Parameters
        ----------
        dataframe : DataFrame
            The data to retrieve the metric from.

        Returns
        -------
        float
            The mean column completeness metric.
        """
        return ((~dataframe.isna()).sum()/dataframe.shape[0]).mean()

    def _get_mean_row_completness(self, dataframe: pd.DataFrame) -> float:
        """
        Calculate the mean row completeness metric.

        Parameters
        ----------
        dataframe : DataFrame
            The data to retrieve the metric from.

        Returns
        -------
        float
            The mean row completeness metric.
        """
        return ((~dataframe.isna()).sum(axis=1)/dataframe.shape[1]).mean()

    def describe(self, dataframe: pd.DataFrame, output_file: str = None, version: str = None) -> Dict:
        """
        Generate description of the dataset and measurements assessing the quality/integrity of the dataset.

        Parameters
        ----------
        dataframe : DataFrame
            The data to describe.
        output_file : str (Optional)
            The path and filename where the description will be written to.
        version : str (Optional)
            The version name of the dataset.

        Returns
        -------
        Dict
            Dictionary containing the description of the dataset and measurements assessing the quality/integrity of the dataset.
        """
        # Version and data source information is based on self.dataset since that's all that's stored
        # Columns, date ranges, and metrics are based on the dataframe input parameter.
        if 'trans_date_trans_time' in dataframe.columns:
            date_ranges = (dataframe['trans_date_trans_time'].min().strftime(
                '%Y-%m-%d_%H-%M-%S_%f'), dataframe['trans_date_trans_time'].max().strftime('%Y-%m-%d_%H-%M-%S_%f'))
        else:
            date_ranges = []

        info = {
            'version': version if version is not None else self.version,
            'data_sources': self.data_source_names,
            'column_names': list(dataframe.columns),
            'column_types': dataframe.dtypes.apply(lambda x: x.name).to_dict(),
            'date_ranges': date_ranges,
            'num_data_sources': len(self.data_sources),
            'transformations': self.transformations,
            'measures': {
                'num_samples': dataframe.shape[0],
                'num_columns': dataframe.shape[1],
                'mean_time_since_transaction': f'{self._get_mean_time_since_transaction(dataframe)} days',
                'table_completeness': self._get_table_completness(dataframe),
                'mean_col_completness': self._get_mean_col_completness(dataframe),
                'mean_row_completness': self._get_mean_row_completness(dataframe)
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
    csv = DataEngineering('./data/transactions_0.csv')
    csv.clean_missing_values()
    csv.remove_duplicates()
    csv.standardize_dates('trans_date_trans_time')
    csv.resolve_anomalous_dates('trans_date_trans_time')
    csv_data = csv.get_dataset()

    testDir = './data/my_tests/'
    # csv = Dataset('./data/transactions_0.csv')
    csv = Dataset(csv_data)
    print(csv.get_data_source())
    csv.load(f'{testDir}csv_to_json', '.json')
    randSamp = csv.sample(sampling_method='random', N=5, seed=1)
    print(randSamp.head(10))
    stratSamp = csv.sample(sampling_method='stratified',
                           N=1, stratification_factors=['category'])
    print(stratSamp.head(15))
    print(csv.describe(csv.get_dataset(), f'{testDir}csvLogs'))


if __name__ == '__main__':
    test()
