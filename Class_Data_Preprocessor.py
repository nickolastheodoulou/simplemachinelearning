import pandas as pd
from sklearn.model_selection import train_test_split
# import fancyimpute as fi

from scipy.special import boxcox

from Class_Data_Explorer import DataExplorer


class DataPreprocessor(DataExplorer):
    def __init__(self, train_data_set, test_data_set):
        super().__init__(train_data_set, test_data_set)

    ####################################################################################################################
    # Use only when train and test are in the same data set

    def split_data_set_if_test_not_split(self, target, my_test_size, seed):
        # set attributes to all other columns in the data_set
        attribute_matrix = self._train_data_set.loc[:, self._train_data_set.columns != target]
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(attribute_matrix,
                                                                                    self._train_data_set[target],
                                                                                    test_size=my_test_size,
                                                                                    random_state=seed)

    ####################################################################################################################

    def split_data_data_set_X_data_set_y(self, target):
        target_column = self._train_data_set.columns.get_loc(target)  # finds the target column by name
        # updates the data frame train_Y
        self._data_set_y = pd.DataFrame(data={target: self._train_data_set.iloc[:, target_column]})
        # drops the first column of the train set as it has been moved
        self._data_set_X = self._train_data_set.drop(self._train_data_set.columns[target_column], axis=1)

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_less_y_attribute_greater_x(self, target, y, attribute, x):
        self._train_data_set = self._train_data_set.drop(self._train_data_set[(self._train_data_set[attribute] > x) &
                                                         (self._train_data_set[target] < y)].index)

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_less_y_attribute_less_x(self, target, y, attribute, x):
        self._train_data_set = self._train_data_set.drop(self._train_data_set[(self._train_data_set[attribute] < x) &
                                                         (self._train_data_set[target] < y)].index)

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_greater_y_attribute_greater_x(self, target, y, attribute, x):
        self._train_data_set = self._train_data_set.drop(self._train_data_set[(self._train_data_set[attribute] > x) &
                                                                              (self._train_data_set[target] > y)].index)

    # Deleting outliers lower and to the right of the trend
    def drop_outliers_target_greater_y_attribute_less_x(self, target, y, attribute, x):
        self._train_data_set = self._train_data_set.drop(self._train_data_set[(self._train_data_set[attribute] < x) &
                                                                              (self._train_data_set[target] > y)].index)

    #  method that drops all the rows with missing data. This is not recommended to be used at all but is used to
    #  test how accurate a simple KNN algorithm is
    def drop_all_na(self):
        self._train_data_set = self._train_data_set.dropna()
        self._test_data_set = self._test_data_set.dropna()

    def drop_attribute(self, attribute):
        self._train_data_set = self._train_data_set.drop(columns=[attribute])
        self._test_data_set = self._test_data_set.drop(columns=[attribute])

    def box_cox_trans_attribute(self, attribute, lamda):  # boxcox transformation of an attribute in train_x
        self._train_data_set[attribute] = boxcox(self._train_data_set[attribute], lamda)
        self._test_data_set[attribute] = boxcox(self._test_data_set[attribute], lamda)

    def box_cox_target(self, lamda):
        self._y_train = boxcox(self._y_train, lamda)

    def normalise_attribute(self, attribute):  # normalises all column of an attribute
        self._train_data_set[attribute] = (self._train_data_set[attribute] - self._train_data_set[attribute].mean()) / \
                                          self._train_data_set[attribute].std()
        self._test_data_set[attribute] = (self._test_data_set[attribute] - self._train_data_set[attribute].mean()) / \
                                          self._train_data_set[attribute].std()

    #  method that one hot encodes a column
    def one_hot_encode_attribute(self, attribute):
        #  define the data set as the original data set combined with the one hot encoded column of the inputted
        # attribute

        # concat adds the new columns to the data set
        # prefix adds the string attribute to the column head
        self._train_data_set = pd.concat([self._train_data_set, pd.get_dummies(self._train_data_set[attribute], prefix=attribute)],
                                         axis=1, sort=False)
        #  drops the column that has the sting value of the attribute to be one hot encoded
        self._train_data_set = self._train_data_set.drop(columns=[attribute])

        self._test_data_set = pd.concat(
            [self._test_data_set, pd.get_dummies(self._test_data_set[attribute], prefix=attribute)],
            axis=1, sort=False)
        self._test_data_set = self._test_data_set.drop(columns=[attribute])

    def impute_mode(self, attribute):
        self._train_data_set[attribute] = self._train_data_set[attribute].fillna(
            self._train_data_set[attribute].mode()[0])

        self._test_data_set[attribute] = self._train_data_set[attribute].fillna(
            self._test_data_set[attribute].mode()[0])

    def impute_median(self, attribute):
        self._train_data_set[attribute] = self._train_data_set[attribute].fillna(
            self._train_data_set[attribute].median())

        self._test_data_set[attribute] = self._train_data_set[attribute].fillna(
            self._test_data_set[attribute].median())

    def impute_mean(self, attribute):
        self._train_data_set[attribute] = self._train_data_set[attribute].fillna(self._train_data_set[attribute].mean())

        self._test_data_set[attribute] = self._test_data_set[attribute].fillna(self._train_data_set[attribute].mean())

    def impute_none(self, attribute):
        self._train_data_set[attribute] = self._train_data_set[attribute].fillna("None")
        self._test_data_set[attribute] = self._test_data_set[attribute].fillna("None")

    #  can also apply this to the test data set without any data leakage!
    def impute_zero(self, attribute):  # fill na with 0
        self._train_data_set[attribute] = self._train_data_set[attribute].fillna(0)
        self._test_data_set[attribute] = self._test_data_set[attribute].fillna(0)

    def switch_na_to_median_other_attribute(self, attribute, second_discrete_attribute):
        # fill in the missing value by grouping by second_discrete_attribute and findin the mean of each group and
        # assigning the missing value to this)
        self._train_data_set[attribute] = self._train_data_set[attribute].fillna(self._train_data_set.groupby(
            second_discrete_attribute)[attribute].mean()[0])

        #  apply to test_X by using the median of train_X to prevent data leakage
        self._test_data_set[attribute] = self._test_data_set[attribute].fillna(
            self._train_data_set.groupby(second_discrete_attribute)[attribute].mean()[0])

    def convert_attribute_to_categorical(self, attribute):
        self._train_data_set[attribute] = self._train_data_set[attribute].astype(str)
        self._test_data_set[attribute] = self._test_data_set[attribute].astype(str)

    # imputes the missing attributes using KNN from fancy impute using the 3 closes complete columns
    # found to be extremely ineffienct hence not used in final model
    '''
    def impute_knn(self, number_of_nearest_neighbours):
        knn_impute = fi.KNN(k=number_of_nearest_neighbours).complete(self._data_set)
        self._data_set = pd.DataFrame(knn_impute, columns=self._data_set.columns.copy())
    '''

    ####################################################################################################################
    # specific to car insurance
    def new_column_infinite_credit_score(self):
        # create a list of the indices of the credit score with a score of 9999
        credit_score_9999_index = self._train_data_set[self._train_data_set['Credit_Score'] == 9999].index.tolist()

        # create a new column for the credit scores with 9999 so they can be put into a different attribute
        self._train_data_set['Infinite_Credit_Score'] = 0

        for i in credit_score_9999_index:
            # set the new column values to 1 (one hot encoding)
            self._train_data_set['Infinite_Credit_Score'].values[i] = 1
            # drop the credit score of 9999 from the attribute Credit_Score
            self._train_data_set['Credit_Score'].values[i] = 0

    def impute_price(self):
        # create a list of the index of the missing values in the price attribute
        price_missing_value_index = self._train_data_set[self._train_data_set['Price'].isnull()].index.tolist()

        # loop through the missing value index for price
        for i in price_missing_value_index:
            # if the value for tax in the same column as price is greater than 34
            if self._train_data_set['Tax'].values[i] > 33:
                # set the value for price equal to ten times the value of tax in the same column
                self._train_data_set['Price'].values[i] = self._train_data_set['Tax'].values[i] * 10
            else:
                # else set the price to 5 times the tax
                self._train_data_set['Price'].values[i] = self._train_data_set['Tax'].values[i] * 5

        print('The number of price values imputed is ', len(price_missing_value_index))

    def impute_tax(self):
        # create a list of the index of the missing values in the tax attribute
        tax_missing_value_index = self._train_data_set[self._train_data_set['Tax'].isnull()].index.tolist()
        # loop through the missing value index for price
        for i in tax_missing_value_index:
            # if the value for tax in the same column as price is greater than 34
            if self._train_data_set['Price'].values[i] > 330:
                # set the value for price equal to ten times the value of tax in the same column
                self._train_data_set['Tax'].values[i] = self._train_data_set['Price'].values[i] * 0.1
            else:
                # else set the price to 5 times the tax
                self._train_data_set['Tax'].values[i] = self._train_data_set['Price'].values[i] * 0.05

        print('The number of tax values imputed is ', len(tax_missing_value_index))

    #  add the day of the week as a new column, could re-write to already be one hot encoded
    def add_day_of_week_attribute(self):
        # convert type of column date form object to datetime64
        self._train_data_set['Date'] = pd.to_datetime(self._train_data_set['Date'], infer_datetime_format=True)
        # add new column named days_of_the_week that has the day of the week
        self._train_data_set = self._train_data_set.assign(
            days_of_the_week=self._train_data_set['Date'].dt.weekday_name)

    ####################################################################################################################
