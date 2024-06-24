from typing import Dict
import pandas as pd
import sys
import json
from datetime import datetime


class DataEngineering():
    """
    This is a class for loading, cleaning, transforming, and validation CSV files of credit card transaction data.

    Attributes:
        dataset : dataframe
            The transaction data.
        transformations : list
            The transformations that have been run on the dataset.
    """

    def __init__(self, filename: str):
        """
        Initialize the instance of the DataEngineering class.

        Parameters
        ----------
        filename : string
            The name of the csv or parquet file of transaction data to be read.

        Returns
        -------
        None
        """
        # Initialize the dataset Load the dataset
        self.dataset = self.load_dataset(filename)
        # Rename the index column (this is specifically for transactions_0.csv from Assignment 2, and may be edited or removed later)
        self.dataset.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
        # Initialize the list of transformations that have been run on the dataset
        self.transformations = []

    # 1. Data Extraction & Summary

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load the data from the given file.

        Parameters
        ----------
        filename : string
            The name of the CSV, json, or parquet file of transaction data to be read.

        Returns
        -------
        dataframe
            The data read from the given filename.
        """
        fileType = filename.split('.')[-1].lower()
        if fileType == 'csv':
            return pd.read_csv(filename)
        elif fileType == 'parquet':
            return pd.read_parquet(filename, 'pyarrow')
        elif fileType == 'json':
            return pd.read_json(filename)
        else:
            raise ValueError(
                'File must be of type .csv, .json, or .parquet')

    def describe(self, N: int) -> Dict:
        """
        Print the first N rows and descriptive information about the data.

        Parameters
        ----------
        N : dictionary
            The number of rows of data to be printed.

        Returns
        -------
        dictionary
            The descriptive information about the data.
        """
        # Print data rows
        print(f'First {N} Rows:\n{self.dataset.head(N)}\n')
        # Put descriptive information in a dictionary
        info = {
            'total number of rows': self.dataset.shape[0],
            'column names': list(self.dataset.columns),
            'type of columns': self.dataset.dtypes.apply(lambda x: x.name).to_dict(),
            'total number of columns': self.dataset.shape[1],
            'transformation history': self.transformations
            # Other description of data to potentially be added later
        }
        # Print the info dictionary nicely
        print('Additional Information:')
        print(json.dumps(info, indent=4))
        # Return the information dictionary
        return info

    # 2. Data Cleaning Pipeline

    def clean_missing_values(self) -> None:
        """
        Clean missing values by removing rows that are missing any required data (cc_num, amt, or trans_num), and filling other missing values with default values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Remove rows that are missing required data (cc_num, amt, trans_num)
        self.dataset.dropna(
            subset=['cc_num', 'amt', 'trans_num'], axis=0, inplace=True)
        # Define default values to use to fill missing values in each column
        values = {
            'trans_date_trans_time': '1900-01-01 00:00:00',
            'merchant': 'unknown',
            'category': 'unknown',
            'first': 'unknown',
            'last': 'unknown',
            'sex': 'U',
            'street': 'unknown',
            'city': 'unknown',
            'state': 'UN',
            'zip': 0,
            'lat': 0,
            'long': 0,
            'city_pop': 0,
            'job': 'unknown',
            'dob': '1900-01-01',
            'unix_time': 0,
            'merch_lat': 0,
            'merch_long': 0,
            'is_fraud': -1,
        }
        # Fill in missing values using the default values defined above
        self.dataset.fillna(value=values, inplace=True)
        # Specify column types (this ensure strings and categories are typed correctly instead of 'object')
        self.dataset['merchant'] = self.dataset['merchant'].astype('string')
        self.dataset['category'] = self.dataset['category'].str.lower().astype(
            'category')
        self.dataset['first'] = self.dataset['first'].astype('string')
        self.dataset['last'] = self.dataset['last'].astype('string')
        self.dataset['sex'] = self.dataset['sex'].astype('category')
        self.dataset['street'] = self.dataset['street'].astype('string')
        self.dataset['city'] = self.dataset['city'].astype('string')
        self.dataset['state'] = self.dataset['state'].astype('category')
        self.dataset['zip'] = self.dataset['zip'].astype('int64')
        self.dataset['job'] = self.dataset['job'].astype('string')
        self.dataset['trans_num'] = self.dataset['trans_num'].astype('string')
        self.dataset['is_fraud'] = self.dataset['is_fraud'].astype('int64')
        # Convert credit card number to string (to avoid int overflow for long card numbers)
        self.dataset['cc_num'] = self.dataset['cc_num'].astype('string')
        # Track that this transformation occurred
        self.transformations.append('clean_missing_values')

    def remove_duplicates(self) -> None:
        """
        Remove duplicate transactions (rows with the same trans_num identifier), keeping the first one.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.dataset.drop_duplicates(subset=['trans_num'], inplace=True)
        # Track that this transformation occurred
        self.transformations.append('remove_duplicates')

    def standardize_dates(self, column_name: str) -> None:
        """
        Convert the values in the given column to a standardized date format.

        Parameters
        ----------
        column_name : string
            The name of the column containing the values to standadize the date format for.

        Returns
        -------
        None
        """
        # Return if the given column name doesn't exist or the column type is not a string or date
        if column_name not in self.dataset.columns or self.dataset.dtypes[column_name].name not in ['object', 'string', 'datetime64[ns]']:
            print(
                f'Column "{column_name}" not found or not string or date type.')
            return
        # Convert column values to datetime
        self.dataset[column_name] = pd.to_datetime(
            self.dataset[column_name], format='mixed', dayfirst=True)
        # Track that this transformation occurred
        self.transformations.append(f'standardize_dates_{column_name}')

    def trim_spaces(self, column_name: str) -> None:
        """
        Remove leading and trailing whitespaces from string values in the given column.

        Parameters
        ----------
        column_name : string
            The name of the column containing the values to trim spaces from.

        Returns
        -------
        None
        """
        # Return if the given column name doesn't exist or the column type is not string (object or category)
        if column_name not in self.dataset.columns or self.dataset.dtypes[column_name].name not in ['object', 'string', 'category']:
            print(f'Column "{column_name}" not found or not string type.')
            return
        # Remove leading and trailing whitespace from values
        self.dataset[column_name] = self.dataset[column_name].str.strip()
        # Track that this transformation occurred
        self.transformations.append(f'trim_spaces_{column_name}')

    def resolve_anomalous_dates(self, column_name: str) -> None:
        """
        Resolve anomolies in date time data in the given column.

        Parameters
        ----------
        column_name : string
            The name of the column containing the date values to resolve.

        Returns
        -------
        None
        """
        # Return if the given column name doesn't exist
        if column_name not in self.dataset.columns:
            print(f'Column "{column_name}" not found.')
            return
        # Put dates into standard format first
        self.standardize_dates(column_name)
        # Return if the given column is not a date
        if self.dataset.dtypes[column_name].name != 'datetime64[ns]':
            print(f'Column "{column_name}" is not date type.')
            return
        # If date is in the future, assume it's wrong and handle it the same as missing/unknown data by applying the same default value used in self.clean_missing_values()
        today = datetime.today()
        self.dataset.loc[self.dataset[column_name] > today,
                         column_name] = pd.Timestamp('1900-01-01 00:00:00')
        # Track that this transformation occurred
        self.transformations.append(f'resolve_anomalous_dates_{column_name}')

    # 3. Data Transformation

    def expand_dates(self, column_name: str) -> pd.DataFrame:
        """
        Find year, month, day_of_week, and hour_of_day from the dates in the given column, and append these values to the dataset in new columns.

        Parameters
        ----------
        column_name : string
            The name of the column containing the values to generate the new columns from.

        Returns
        -------
        dataframe
            The dataset with year, month, day_of_week, and hour_of_day columns appended. 
        """
        # Return if the given column name doesn't exist
        if column_name not in self.dataset.columns:
            print(f'Column "{column_name}" not found.')
            return
        # Put dates into standard format first
        self.standardize_dates(column_name)
        # Return if the given column is not a date
        if self.dataset.dtypes[column_name].name != 'datetime64[ns]':
            print(f'Column "{column_name}" is not date type.')
            return
        # Generate year column
        self.dataset[f'year_{column_name}'] = self.dataset[column_name].dt.year
        # Generate month column
        self.dataset[f'month_{column_name}'] = self.dataset[column_name].dt.month
        # Generate day_of_week column
        self.dataset[f'day_of_week_{column_name}'] = self.dataset[column_name].dt.day_of_week
        # Generate hour_of_day column
        self.dataset[f'hour_of_day_{column_name}'] = self.dataset[column_name].dt.hour
        # Track that this transformation occurred
        self.transformations.append(f'expand_dates_{column_name}')
        # Return the dataframe
        return self.dataset

    def categorize_transactions(self, low: float, medium: float, high: float) -> pd.DataFrame:
        """
        Categorize the transaction amount as low, medium, or high based on the given quartile values, and append this category to the dataset in a new column.
        Note: the assignment description was unclear regarding what the low, medium, high inputs represent, so I defined them as below.

        Parameters
        ----------
        low : float
            The upper bound for the quantile value at which the 'low' category ends (e.g. 0.25).
        medium : float
            The center for the quantile value at which the 'medium' category is centered (e.g. 0.5). Note: this is unused due to how qcut works and the fact that we're dividing into 3 categories.
        high : float
            The lower bound for the quantile value at which the 'high' category begins (e.g. 0.75).

        Returns
        -------
        dataframe
            The dataset with amt_category column appended. 
        """
        # First clean the data
        self.clean_missing_values()
        # Initialize empty amt_category column
        colName = f'amt_category_{low}-{medium}-{high}'
        # Use qcut to discritize amount column into categorical bins.
        # We must use 0 and 1 as lower/upper quantile bounds to ensure all data is categorized and avoid NA values.
        # Since we have 3 specicified labels ('low', 'medium', 'high') and the upper/lower quantile bounds 0 and 1,
        #   then qcut() requires only 2 other quantile values. By providing low and high along with our upper and lower bounds 0 and 1,
        #   then we'll end up with 3 categories, where medium is everything between low and high.
        #   The medium input value cannot be used due to how qcut() works since we have 3 category labels.
        #   e.g. low=0.25 and high=0.75, then low is < 0.25, medium is between 0.25 and 0.75, and high is > 0.75
        categories = pd.qcut(self.dataset['amt'], [0, low, high, 1], labels=[
                             'low', 'medium', 'high'])
        self.dataset[colName] = categories
        self.transformations.append('categorize_transactions')
        # Return the dataframe
        return self.dataset

    # 4. Data Validation

    def range_checks(self) -> bool:
        """
        Validate whether numerical columns fall within expected bounds.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Boolean indicating whether all values in all numerical columns fall within expected bounds. 
        """
        # Define expected bounds for numerical columns (assuming known columns defined in Module 2)
        expected_bounds = {
            'Index': [0, sys.maxsize],
            'amt': [0, sys.maxsize],
            'zip': [0, 99950],  # 0 = unknown, others = 00001 to 99950
            'lat': [-90, 90],
            'long': [-180, 180],
            'city_pop': [0, sys.maxsize],
            # Up to current time
            'unix_time': [-sys.maxsize, datetime.today().timestamp()],
            'merch_lat': [-90, 90],
            'merch_long': [-180, 180],
            'is_fraud': [-1, 1],  # -1 = unknown, 0 = not fraud, 1 = fraud,
            'day_of_week': [0, 6],
            'hour_of_day': [0, 23]
        }
        # Get numerical data
        numerical_data = self.dataset.select_dtypes(include='number')
        # For each numerical column, check if all values fall within the expected bounds
        for colName in list(numerical_data.columns):
            # Get column values
            colData = numerical_data[colName]
            colKey = 'day_of_week' if 'day_of_week' in colName else 'hour_of_day' if 'hour_of_day' in colName else colName
            bounds = expected_bounds.get(colKey, [-sys.maxsize, sys.maxsize])
            # Check if data in the current column is within valid range
            colValid = ((colData >= bounds[0]) & (colData <= bounds[1])).all()
            # Return False if we've found an invalid column
            if colValid == False:
                print(f'{colName} range_checks failed.')
                return False
        # Return True since we've successfully looped through all numerical columns without returning False
        return True

    def null_checks(self) -> bool:
        """
        Validate whether all essential columns have no null values. Since null values are filled during cleaning, all columns should not have nulls.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Boolean indicating whether all values in all essential columns are not null. 
        """
        # For each column, check if all values are not null
        for colName in list(self.dataset.columns):
            # Check if data in the current column doesn't have any null values
            colValid = self.dataset[colName].notna().all()
            # Return False if we've found an invalid column
            if colValid == False:
                print(f'{colName} null_checks failed.')
                return False
        # Return True since we've successfully looped through all columns without returning False
        return True

    def type_validation(self) -> bool:
        """
        Validate whether all data types are consistent with expected format.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Boolean indicating whether all data types are consistent with expected format. 
        """
        # Define expected type for each known column
        expected_type = {
            'Index': 'int64',
            'trans_date_trans_time': 'datetime64[ns]',
            'cc_num': 'string',
            'merchant': 'string',
            'category': 'category',
            'amt': 'float64',
            'first': 'string',
            'last': 'string',
            'sex': 'category',
            'street': 'string',
            'city': 'string',
            'state': 'category',
            'zip': 'int64',
            'lat': 'float64',
            'long': 'float64',
            'city_pop': 'float64',
            'job': 'string',
            'dob': 'datetime64[ns]',
            'trans_num': 'string',
            'unix_time': 'float64',
            'merch_lat': 'float64',
            'merch_long': 'float64',
            'is_fraud': 'int64',
            'day_of_week': 'int32',
            'hour_of_day': 'int32',
            'amt_category': 'category',
        }
        # For each column, check if the type matches the expected type
        for colName in list(self.dataset.columns):
            # Check if the current column type matches the expected type
            colKey = 'day_of_week' if 'day_of_week' in colName else 'hour_of_day' if 'hour_of_day' in colName else 'amt_category' if 'amt_category' in colName else colName
            expected = expected_type.get(colKey)
            colValid = self.dataset.dtypes[colName] == expected
            # Return False if we've found an invalid column
            if colValid == False:
                print(f'{colName} type_validation failed.')
                return False
        # Return True since we've successfully looped through all columns without returning False
        return True

    def uniqueness_validation(self) -> bool:
        """
        Validate whether all transactions are unique in expected columns. The expected unique columns are Index and trans_num.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Boolean indicating whether all transactions are unique in expected columns. 
        """
        uniqueColumns = ['Index', 'trans_num']
        # For each  expected unique column, check if the values are unique
        for colName in uniqueColumns:
            # Check if values in the current column are unique
            colValid = self.dataset.shape[0] == len(
                self.dataset[colName].unique())
            # Return False if we've found an invalid column
            if colValid == False:
                print(f'{colName} uniqueness_validation failed.')
                return False
        # Return True since we've successfully looped through all specified columns without returning False
        return True

    def historical_data_consistency(self) -> bool:
        """
        Validate whether new data entries are consistent with historical trends or benchmarks in transaction volumes or amounts.
        Note: The assignment description didn't specify any function inputs or the format of the new data entries or what trends to use.
        Therefore, I considered "new" data the rows where is_fraud is unknown (-1) and trends to be the transaction amount.
        I check that the mean and 25%, 50%, and 75% quartile values of the new transaction amounts are within 10% of those values for the historical transaction amounts.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Boolean indicating whether new data entries are consistent with historical trends or benchmarks in transaction volumes or amounts. 
        """
        # Get new data (where is_fraud is unknown, -1)
        new_data = self.dataset.loc[self.dataset['is_fraud'] == -1]
        # Get historical data (where is_fraud is known, 0 or 1)
        hist_data = self.dataset.loc[self.dataset['is_fraud'] >= 0]
        # Check that new_data mean is within 10% of hist_data mean
        histMean = hist_data['amt'].mean()
        newMean = new_data['amt'].mean()
        consistent_means = abs(histMean-newMean) <= 0.1*histMean
        # Check that new_data 25%, 50% and 75% quartiles are within 10% of hist_data quartiles
        histQuartiles = hist_data['amt'].quantile([0.25, 0.5, 0.75])
        newQuartiles = new_data['amt'].quantile([0.25, 0.5, 0.75])
        consistent_quartiles = (
            abs(histQuartiles-newQuartiles) < 0.1*histQuartiles).all()
        # Return boolean indicating whether all values are consistent
        return consistent_means & consistent_quartiles

    def categorical_data_validation(self) -> bool:
        """
        Validate whether all categorical values are in the expected list of values for the category.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Boolean indicating whether all categorical values are in the expected list of values for the category. 
        """
        # Define approved list of values for each knwon categorical field
        approved_values = {
            # 'category' list includes 'unknown' from cleaning step
            'category': ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport',
                         'misc_pos', 'grocery_net', 'shopping_pos', 'shopping_net',
                         'food_dining', 'personal_care', 'health_fitness', 'travel',
                         'kids_pets', 'home', 'unknown'],
            # 'sex' list inlcudes 'U' for unknown from cleaining step
            'sex': ['F', 'M', 'U'],
            # 'state' list inlcudes 'UN' for unknown from cleaining step, and DC ans US territories
            'state': ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA",
                      "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO",
                      "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK",
                      "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI",
                      "WV", "WY", "DC", "AS", "GU", "MP", "PR", "VI", "UN"],
            'amt_category': ['low', 'medium', 'high'],
        }
        # Get categorical data
        categorical_data = self.dataset.select_dtypes(include='category')
        # For each column of categorical data, check that all values are in the approved list for the category
        for colName in list(categorical_data.columns):
            # Get the approved values for the current column's category
            colKey = 'amt_category' if 'amt_category' in colName else colName
            approvedVals = approved_values.get(colKey)
            # Check whether all values in the current column are one of the approved values
            colValid = categorical_data[colName].isin(approvedVals).all()
            # Return False if we've found an invalid column
            if colValid == False:
                print(f'{colName} categorical_data_validation failed.')
                return False
        # Return True since we've successfully looped through all columns without returning False
        return True

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
    """
    Function to briefly test all methods in the DataEngineering class.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    de = DataEngineering('./data/transactions_0.csv')
    de.describe(10)
    print('Cleaning and transforming data...')
    de.clean_missing_values()
    de.remove_duplicates()
    de.trim_spaces('merchant')
    de.standardize_dates('trans_date_trans_time')
    de.standardize_dates('dob')
    de.resolve_anomalous_dates('trans_date_trans_time')
    de.resolve_anomalous_dates('dob')
    de.expand_dates('trans_date_trans_time')
    de.expand_dates('dob')
    de.categorize_transactions(0.25, 0.5, 0.75)
    de.describe(10)
    print('Validating data...')
    print(f'range_checks passed: {de.range_checks()}')
    print(f'null_checks passed: {de.null_checks()}')
    print(f'type_validation passed: {de.type_validation()}')
    print(f'uniqueness_validation passed: {de.uniqueness_validation()}')
    print(
        f'historical_data_consistency passed: {de.historical_data_consistency()}')
    print(
        f'categorical_data_validation passed: {de.categorical_data_validation()}')


if __name__ == '__main__':
    test()
