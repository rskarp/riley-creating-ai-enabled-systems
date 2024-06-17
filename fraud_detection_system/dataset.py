from datetime import datetime
from typing import Union, List, Dict, Tuple
import pandas as pd
import json

from data_engineering import DataEngineering


class Dataset():
    """
    This is a class for constructing a dataset from raw credit card transaction data.

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
        Initialize the instance of the Dataset class.

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
        if hasattr(Dataset, 'dataset'):
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

    def _stratified_sample(self, N: int, strat_factors: List[str]) -> pd.DataFrame:
        """
        Select samples from self.dataset.

        Parameters
        ----------
        N : int
            The number of rows to return per stratum if using stratified sampling.
        stratification_factors : List[str]
            The names of columns to be used as stratification factors.

        Returns
        -------
        DataFrame
            The samples selected as a subset from self.dataset.
        """
        return self.dataset.groupby([f.lower() for f in strat_factors], group_keys=False).apply(lambda x: x.sample(min(len(x), N)))

    def sample(self, *args, **kwargs) -> pd.DataFrame:
        """
        Select samples from self.dataset.

        Parameters
        ----------
        N : int
            The number of rows to return per stratum if using stratified sampling, or total for random sampling.
        sampling_method : str
            The sampling method to use: random or stratified.
        stratification_factors : List[str]
            The names of columns to be used as stratification factors.
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
        strat_factors = kwargs.get('stratification_factors', [])
        if sampMethod == 'random':
            return self._random_sample(N, seed)
        elif sampMethod == 'stratified':
            return self._stratified_sample(N, strat_factors)

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
        return dataframe.isna().sum().sum()/dataframe.size

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
        return (dataframe.isna().sum()/dataframe.shape[0]).mean()

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
        return (dataframe.isna().sum(axis=1)/dataframe.shape[1]).mean()

    def describe(self, dataframe: pd.DataFrame, output_file: str = None) -> Dict:
        """
        Generate description of the dataset and measurements assessing the quality/integrity of the dataset.

        Parameters
        ----------
        dataframe : DataFrame
            The data to describe.
        output_file : str (Optional)
            The path and filename where the description will be written to.

        Returns
        -------
        Dict
            Dictionary containing the description of the dataset and measurements assessing the quality/integrity of the dataset.
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
                'num_samples': dataframe.shape[0],
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
        return self.dataset


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
