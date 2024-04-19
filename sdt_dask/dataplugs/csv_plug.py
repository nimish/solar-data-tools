"""Class for locally saved files from the Cassandra DB"""

import pandas as pd
from solardatatools.time_axis_manipulation import make_time_series
from sdt_dask.dataplugs.dataplug import DataPlug


class LocalFiles(DataPlug):
    """Dataplug class for retrieving data from some source. It's recommended
    that user-created dataplug inherit from this class to ensure compatibility.

    The initialization argument for each class will be different depending on
    the source. The main requirement is to keep the ``Dataplug.get_data`` method,
    and make sure the args and returns as defined here.
    """

    def __init__(self, path_to_files, ext=".csv"):
        self.path = path_to_files
        self.ext = ext.lower()

    def _read_file(self, filename: str):
        print(f"Loading file {filename}...")
        file = self.path + filename + self.ext
        if self.ext == ".csv":
            self.df = pd.read_csv(file)
        else:
            raise ValueError("Only .csv files are supported")

    def _clean_data(self):
        # Convert index from int to datetime object
        self.df, _ = make_time_series(self.df)

    def get_data(self, key: tuple[str]) -> pd.DataFrame:
        """This is the main function that the Dask tool will interact with.
        Users should keep the args and returns as defined here when writing
        their custom dataplugs.

        :param key: Filename (without the extension suffix)--typically designating
            a unique set of historical power generation measurements
        :return: Returns a pandas DataFrame with a timestamp column and
            a power column
        """
        self._read_file(key[0])
        self._clean_data()

        return self.df
