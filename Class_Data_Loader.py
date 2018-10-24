from sklearn.utils import shuffle
import pandas as pd


class DataLoader:  # class that stores the data set as an object. The purpose of this class is to
    def __init__(self, train_data_set, test_data_set):  # initialise the object with the data_set
        self._x_train = 0
        self._x_test = 0
        self._y_train = 0
        self._y_test = 0
        self._data_set_y = 0
        self._data_set_X = 0

        self._train_data_set = train_data_set  # data set
        self._test_data_set = test_data_set
        self._test_y_id = 0  # id column later used to add to Pred_Y if the index_column is dropped
        # the underscore means that the members are protected
        print(self, 'created')

    #  method that shuffles the data set
    def shuffle_data_set(self):
        # shuffle using sklearn.utils, seed set to 0 to get the same shuffle each time to test model
        self._train_data_set = shuffle(self._train_data_set, random_state=0)

    #  function that creates a new string column by combining two other columns
    def combine_columns(self, new_column_name, first_column_to_combine, second_column_to_combine):
        self._train_data_set[new_column_name] = self._train_data_set[first_column_to_combine].map(str) + " " + \
                                                self._train_data_set[second_column_to_combine].map(str)

    #  function that drops the first column of both train_X and test_X
    def index_column_drop_and_move_to_pred_y(self, index_column_label):
        # drops the first column of the train set as the id so that it isn't included in the model
        self._train_data_set = self._train_data_set.drop(self._train_data_set.columns[0], axis=1)
        # define id so that it can be added to pred_y
        self._test_y_id = self._test_data_set[index_column_label]
        # drops the first column of the test set as the id so that it isnt included in the model
        self._test_data_set = self._test_data_set.drop(self._test_data_set.columns[0], axis=1)
        return None

    def move_target_to_train_y(self, target):
        # finds the target column by name passed through from the function
        target_column = self._train_data_set.columns.get_loc(target)
        # updates the data frame train_Y using the index column
        d = {target: self._train_data_set.iloc[:, target_column]}
        self._y_train = pd.DataFrame(data=d)
        # drops the first column of the train set as it has been moved
        self._train_data_set = self._train_data_set.drop(self._train_data_set.columns[target_column], axis=1)
        return None

    def __del__(self):
        print(self, 'destroyed')  # print statement when the destructor is called
