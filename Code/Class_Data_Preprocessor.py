import pandas as pd

from Code.Class_Data_Explorer import DataExplorer


class DataPreprocessor(DataExplorer):
    def __init__(self, data_set):
        super().__init__(data_set)

    #  method that drops all the rows with missing data. This is not recommended to be used at all but is used to
    #  test how accurate a simple KNN algorithm is
    def drop_all_na(self):
        self._data_set = self._data_set.dropna()

    def drop_attribute(self, attribute):
        self._data_set = self._data_set.drop(columns=[attribute])

    def normalise_data(self, target):  # normalises all the data apart by excluding the target
        #  define attributes to normalise that exclude the target
        data_to_normalise_without_target = self._data_set.loc[:, self._data_set.columns != target]
        #  set all the data in the data set object excluding the target to the normalised values
        self._data_set.loc[:, self._data_set.columns != target] = (data_to_normalise_without_target - data_to_normalise_without_target.mean()) / data_to_normalise_without_target.std()