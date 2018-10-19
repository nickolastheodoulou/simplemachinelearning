from scipy.special import boxcox
from sklearn.model_selection import train_test_split
import pandas as pd

from Class_Data_Explorer import DataExplorer


class DataPreprocessor(DataExplorer):
    def __init__(self, data_set):
        super().__init__(data_set)

    #  method that drops all the rows with missing data. This is not recommended to be used at all but is used to
    #  test how accurate a simple KNN algorithm is
    def drop_all_na(self):
        self._data_set = self._data_set.dropna()

    def drop_attribute(self, attribute):
        self._data_set = self._data_set.drop(columns=[attribute])

    def boxcox_trans_attribute(self, attribute, lamda):  # boxcox transformation of an attribute in train_x
        self._data_set[attribute] = boxcox(self._data_set[attribute], lamda)

    def normalise_attribute(self, attribute):  # normalises all column of an attribute
        self._data_set[attribute] = (self._data_set[attribute] - self._data_set[attribute].mean()) / self._data_set[attribute].std()

    #  method that one hot encodes a column
    def one_hot_encode_attribute(self, attribute):
        #  define the data set as the original data set combined with the one hot encoded column of the inputted
        # attribute

        # concat adds the new columns to the data set
        # prefix adds the string attribute to the column head
        self._data_set = pd.concat([self._data_set, pd.get_dummies(self._data_set[attribute], prefix=attribute)],
                                   axis=1, sort=False)

        #  drops the column that has the sting value of the attribute to be one hot encoded
        self._data_set = self._data_set.drop(columns=[attribute])

    #  add the day of the week as a new column, could re-write to already be one hot encoded
    def add_day_of_week_attribute(self):
        # convert type of column date form object to datetime64
        self._data_set['Date'] = pd.to_datetime(self._data_set['Date'], infer_datetime_format=True)
        # add new column named days_of_the_week that has the day of the week
        self._data_set = self._data_set.assign(days_of_the_week=self._data_set['Date'].dt.weekday_name)

        # self._data_set = pd.concat([self._data_set, pd.get_dummies(self._data_set['days_of_the_week'],
        #                                                            prefix='days_of_the_week')], axis=1, sort=False)
        #  self._data_set = self._data_set.drop(columns=['days_of_the_week'])

