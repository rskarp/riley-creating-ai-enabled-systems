from datetime import datetime
from typing import Union, List, Dict, Tuple
import pandas as pd


class Dataset():
    """
    This is a class for constructing a dataset from raw credit card transaction data.

    Attributes:
        dataset : dataframe
            The transaction data.
        transformations : list
            The transformations that have been run on the dataset.
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
        self.dataset = self.extract_data(raw_data)

    # Extract and Load

    def extract_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load data from either a file (.csv, .json, or .parquet) or a pandas.DataFrame.

        Parameters
        ----------
        data : Union[str, pd.DataFrame]
            The .

        Returns
        -------
        DataFrame
            The.
        """
        pass

    def load(self, output_filename: str, format: str = None) -> None:
        """
        Write self.dataset to file.

        Parameters
        ----------
        output_filename : string
            The where the path and filename you intend to write the dataset.
        format : string (optional)
            The format of the output file (accept: .csv, .json, .parquet).

        Returns
        -------
        None
        """
        # it is a good idea to include the version identifier in the output filename.
        pass

    # Description

    def get_data_source(self) -> List[str]:
        """
        indicates from which file the dataset is constructed.

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            The.
        """
        pass

    def set_version(self) -> str:
        """
        Gives the dataset a unique identifier.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The unique identifier for self.dataset.
        """
        pass

    # Sampling

    def sample(self, *args, **kwargs) -> pd.DataFrame:
        """
        Select samples from self.dataset.

        Parameters
        ----------
        Variable

        Returns
        -------
        DataFrame
            The .
        """
        # For best practices, you should consider writing multiple supporting functions to modularize
        #     the structure of this function for readability
        # Consider using *args and **kwargs, to store parameters when constructing your dataset.
        #       It is important to be able to trace how these datasets were constructed.
        pass

    # Measures and Logging

    def describe(self, dataframe: pd.DataFrame, output_file: str = None) -> Dict:
        """
        returns a python dictionary with a description of and measurements assessing the quality/integrity of the dataset.

        Parameters
        ----------
        output_file : str (Optional)
            the path and file name you intend to write this dictionary. (.json  file).

        Returns
        -------
        Dict
            The python dictionary with a description of and measurements assessing the quality/integrity of the dataset.
        """
        # At minimum, you should have a description of the dataset's `version`, `data sources`, `column names`, and `date ranges`
        # Also report quality measures
        #  file will be used by analysts to assess the relevance and usability of the dataset constructed, so you should provide sufficient information about this dataset.
        # good idea to include the version identifier and the date created in the output filename
        basic = {
            'description': {
                'version': str,
                'data sources': List[str,],
                'column names': List[str,],
                # fixed-sized tuple of length 2
                'date ranges': Tuple[datetime, datetime]
                # ...
            },
            'measures': {'metric_1': float, 'metric_2': float}
        }


def test():
    pass


if __name__ == '__main__':
    test()
