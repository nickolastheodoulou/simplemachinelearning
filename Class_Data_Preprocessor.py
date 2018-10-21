import pandas as pd
# import fancyimpute as fi

from scipy.special import boxcox

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

    def impute_price(self):
        # create a list of the index of the missing values in the price attribute
        price_missing_value_index = self._data_set[self._data_set['Price'].isnull()].index.tolist()

        # loop through the missing value index for price
        for i in price_missing_value_index:
            # if the value for tax in the same column as price is greater than 34
            if self._data_set['Tax'].values[i] > 33:
                # set the value for price equal to ten times the value of tax in the same column
                self._data_set['Price'].values[i] = self._data_set['Tax'].values[i] * 10
            else:
                # else set the price to 5 times the tax
                self._data_set['Price'].values[i] = self._data_set['Tax'].values[i] * 5

        print('The number of price values imputed is ', len(price_missing_value_index))

    def impute_tax(self):
        # create a list of the index of the missing values in the tax attribute
        tax_missing_value_index = self._data_set[self._data_set['Tax'].isnull()].index.tolist()
        # loop through the missing value index for price
        for i in tax_missing_value_index:
            # if the value for tax in the same column as price is greater than 34
            if self._data_set['Price'].values[i] > 330:
                # set the value for price equal to ten times the value of tax in the same column
                self._data_set['Tax'].values[i] = self._data_set['Price'].values[i] * 0.1
            else:
                # else set the price to 5 times the tax
                self._data_set['Tax'].values[i] = self._data_set['Price'].values[i] * 0.05

        print('The number of tax values imputed is ', len(tax_missing_value_index))

    def impute_mode(self, attribute):
        self._data_set[attribute] = self._data_set[attribute].fillna(self._data_set[attribute].mode()[0])
        # print('The number of tax values imputed is ', len(tax_missing_value_index))

    def impute_median(self, attribute):
        self._data_set[attribute] = self._data_set[attribute].fillna(self._data_set[attribute].median())

    def impute_mean(self, attribute):
        self._data_set[attribute] = self._data_set[attribute].fillna(self._data_set[attribute].mean())

    # imputes the missing attributes using KNN from fancy impute using the 3 closes complete columns
    # found to be extremely ineffienct hence not used in final model
    '''
    def impute_knn(self, number_of_nearest_neighbours):
        knn_impute = fi.KNN(k=number_of_nearest_neighbours).complete(self._data_set)
        self._data_set = pd.DataFrame(knn_impute, columns=self._data_set.columns.copy())
    '''

    def new_column_infinite_credit_score(self):
        # create a list of the indices of the credit score with a score of 9999
        credit_score_9999_index = self._data_set[self._data_set['Credit_Score'] == 9999].index.tolist()

        # create a new column for the credit scores with 9999 so they can be put into a different attribute
        self._data_set['Infinite_Credit_Score'] = 0

        for i in credit_score_9999_index:
            # set the new column values to 1 (one hot encoding)
            self._data_set['Infinite_Credit_Score'].values[i] = 1
            # drop the credit score of 9999 from the attribute Credit_Score
            self._data_set['Credit_Score'].values[i] = 0
