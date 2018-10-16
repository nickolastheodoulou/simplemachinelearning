import pandas as pd
from Code.Class_Data_Exploration import DataExploration
from scipy.special import boxcox, inv_boxcox


class DataPreprocessing(DataExploration):
    def __init__(self, train_x, test_x):
        super().__init__(train_x, test_x)
        self._train_x_string = 0  # all string attributes of train_X
        self._test_x_string = 0  # all string attributes for test_x
        self._train_x_int_float = 0  # all int and float attributes for train_X
        self._test_x_int_float = 0  # all int and float attributes for test_x

    def boxcox_trans(self, attribute, lamda):  # boxcox transformation of an attribute in train_x
        self._train_x[attribute] = boxcox(self._train_x[attribute], lamda)

    def boxcox_trans_inv(self, attribute, lamda):  # boxcox transformation of an attribute in train_x
        self._train_x[attribute] = inv_boxcox(self._train_x[attribute], lamda)

        # need to come up with better way of doing this!!!

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_less_y_attribute_greater_x(self, target, y, attribute, x):
        self._train_x = self._train_x.drop(self._train_x[(self._train_x[attribute] > x) &
                                                         (self._train_x[target] < y)].index)

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_less_y_attribute_less_x(self, target, y, attribute, x):
        self._train_x = self._train_x.drop(self._train_x[(self._train_x[attribute] < x) &
                                                         (self._train_x[target] < y)].index)

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_greater_y_attribute_greater_x(self, target, y, attribute, x):
        self._train_x = self._train_x.drop(self._train_x[(self._train_x[attribute] > x) &
                                                         (self._train_x[target] > y)].index)

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_greater_y_attribute_less_x(self, target, y, attribute, x):
        self._train_x = self._train_x.drop(self._train_x[(self._train_x[attribute] < x) &
                                                         (self._train_x[target] > y)].index)

    #  some data has na when in reality it isn't missing it should just be none
    #  can also apply this to the test data set without any data leakage!
    def switch_na_to_none(self, attribute):
        self._train_x[attribute] = self._train_x[attribute].fillna("None")
        self._test_x[attribute] = self._test_x[attribute].fillna("None")

    #  can also apply this to the test data set without any data leakage!
    def switch_na_to_zero(self, attribute):  # fill na with 0
        self._train_x[attribute] = self._train_x[attribute].fillna(0)
        self._test_x[attribute] = self._test_x[attribute].fillna(0)

    def switch_na_to_mode(self, attribute):  # fill na with mode
        self._train_x[attribute] = self._train_x[attribute].fillna(self._train_x[attribute].mode()[0])
        #  fill na with mode of train_X to prevent data leakage!
        self._test_x[attribute] = self._test_x[attribute].fillna(self._train_x[attribute].mode()[0])

        # Group by a second discrete attribute and fill in missing value by the median attributes of all the second
        # discrete attribue
    def switch_na_to_median_other_attribute(self, attribute, second_discrete_attribute):
        # fill in the missing value by grouping by second_discrete_attribute and findin the mean of each group and
        # assigning the missing value to this)
        self._train_x[attribute] = self._train_x[attribute].fillna(self._train_x.groupby(second_discrete_attribute)
                                                                   [attribute].mean()[0])
        #  apply to test_X by using the median of train_X to prevent data leakage
        self._test_x[attribute] = self._test_x[attribute].fillna(self._train_x.groupby(second_discrete_attribute)
                                                                 [attribute].mean()[0])

    def drop_attribute_train_and_test(self, attribute):
        self._train_x = self._train_x.drop([attribute], axis=1)
        self._test_x = self._test_x.drop([attribute], axis=1)

    def convert_attribute_to_categorical(self, attribute):
        self._train_x[attribute] = self._train_x[attribute].astype(str)
        self._test_x[attribute] = self._test_x[attribute].astype(str)

    # function that updates the variables: _train_x_string, _test_x_string, _train_x_int_float, _test_x_int_float
    def split_attributes(self):
        # updates the data set variable: _train_x_string that contains all the "object" data types in train
        self._train_x_string = self._train_x.select_dtypes(include=['object']).copy()
        # updates the dataset variable: _test_x_string that contains all the "object" datatypes in test
        self._test_x_string = self._test_x.select_dtypes(include=['object']).copy()
        # updates the dataset variable: _train_x_string that contains all the "object" datatypes in train
        self._train_x_int_float = self._train_x.select_dtypes(exclude=['object']).copy()
        # updates the dataset variable: _test_x_string that contains all the "object" datatypes in test
        self._test_x_int_float = self._test_x.select_dtypes(exclude=['object']).copy()
        return None

    def boxcox_attributes(self, alpha):  # normalises the desired data frames
        self._train_x_int_float = boxcox(self._train_x_int_float, alpha)
        self._test_x_int_float = boxcox(self._test_x_int_float, alpha)

    def normalise_data(self):  # normalises the desired data frames
        self._train_x_int_float = (self._train_x_int_float - self._train_x_int_float.mean()) /\
                                  self._train_x_int_float.std()  # normalise _train_x_int_float using standard score

        self._test_x_int_float = (self._test_x_int_float - self._test_x_int_float.mean()) / self._test_x_int_float.std()
        # train_X_string and _test_x_string are not normalised as there is no point when one hot encoding
        # also shouldn't normalise train_Y

    #  function that performs one_hot_encoding on the variables _train_x_string and _test_x_string then updates the
    # variables
    def one_hot_encoding(self):
        # method to convert all the string attributes into one hot encoded by updating the dataframe from pandas
        self._train_x_string = pd.get_dummies(self._train_x_string)
        # method to convert all the string attributes into one hot encoded
        self._test_x_string = pd.get_dummies(self._test_x_string)
        return None

    #  first finds the missing columns of test_X and fills them with zeros in the correct place
    def combine_string_int_float(self):
        # might run into error if test_X has more columns than train_X in other situations
        # combines the int/float matrices and the string matrices and update them to _train_x and _test_x

        # Get missing columns in the training test
        missing_cols = set(self._train_x_string.columns) - set(self._test_x_string.columns)
        # Add a missing column in test set with default value equal to 0
        for i in missing_cols:
            self._test_x_string[i] = 0
            # Ensure the order of column in the test set is in the same order than in train set
        self._test_x_string = self._test_x_string[self._train_x_string.columns]

        self._train_x = pd.concat([self._train_x_int_float, self._train_x_string], axis=1)  # combines the train_X
        self._test_x = pd.concat([self._test_x_int_float, self._test_x_string], axis=1)

    def export_csv_processed(self):  # exports the cleaned train_X, train_Y and test_X data to a seperate CSV file
        self._train_x.to_csv('Data_Out/train_X_up.csv', index=False)
        self._test_x.to_csv('Data_Out/test_X_up.csv', index=False)
        self._train_y.to_csv('Data_Out/train_Y_up.csv', index=False)