from scipy.special import boxcox
from sklearn.model_selection import train_test_split

from Class_Data_Explorer import DataExplorer


class DataPreprocessor(DataExplorer):
    def __init__(self, data_set):
        super().__init__(data_set)
        self._x_train = 0
        self._x_test = 0
        self._y_train = 0
        self._y_test = 0

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

    def split_data_set_into_train_x_test_x_train_y_test_y(self, target, my_test_size, seed):
        # set attributes to all other columns in the data_set
        attribute_matrix = self._data_set.loc[:, self._data_set.columns != target]
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(attribute_matrix,
                                                                                    self._data_set[target],
                                                                                    test_size=my_test_size,
                                                                                    random_state=seed)
