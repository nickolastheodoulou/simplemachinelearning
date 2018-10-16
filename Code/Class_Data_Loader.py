import pandas as pd


class DataLoader:  # class that creates the data matrix by initializing test_X and train_X from main
    def __init__(self, train_x, test_x):  # initialise the object with the test and train matrices from the CSV file

        self._test_x = test_x  # test dataframe
        self._train_x = train_x  # train dataframe
        self._train_y = 0  # train_Y and test_Y given as arguments as in some instances the data will already be split
        self._test_y = 0
        self._test_y_id = 0  # id column later used to add to Pred_Y if the index_column is dropped
        self.train_x_and_test_x = 0
        # the underscore means that the members are protected
        print(self, 'created')

    def __del__(self):
        print(self, 'destroyed')

    # function that finds the dimensions of both the train and test set and stores them in the object
    def dim_data(self):
        print('The dimensions of train_X is', self._train_x.shape, 'and the dimension of test_X is ',
              self._test_x.shape)

    # function to add train_Y to train_X if it is separate so that only 1 function is made to analyse the data in
    #  train_X,
    def add_train_Y_to_train_X(self):
        #  there is later a function that splits them in the data_preprocessing class
        self._train_x = pd.concat([self._train_x, self._train_y], axis=1)  # combines the train_X

    # function that removes the last column of Train_X and puts it into Train_Y (only called if an index column is in
    # the data sets
    def move_target_to_train_y(self, target):
        # finds the target column by name passed through from the function
        final_column = self._train_x.columns.get_loc(target)
        d = {target: self._train_x.iloc[:, final_column]}  # updates the data frame train_Y using the index column
        self._train_y = pd.DataFrame(data=d)
        # drops the first column of the train set as it has been moved
        self._train_x = self._train_x.drop(self._train_x.columns[final_column], axis=1)
        return None

    # function that removes the last column of Test_X and puts it into Test_Y
    def move_target_to_test_y(self, target):
        final_column = self._test_x.columns.get_loc(target)  # finds the target column by name
        d = {target: self._test_x.iloc[:, final_column]}  # updates the data frame train_Y using the index column
        self._test_y = pd.DataFrame(data=d)  # updates the data frame train_Y
        # drops the first column of the train set as it has been moved
        self._test_x = self._test_x.drop(self._test_x.columns[final_column], axis=1)
        return None

    #  function that drops the first column of both train_X and test_X
    def index_column_drop_and_move_to_pred_y(self, index_column_label):
        # drops the first column of the train set as the id so that it isn't included in the model
        self._train_x = self._train_x.drop(self._train_x.columns[0], axis=1)
        # define id so that it can be added to pred_y
        self._test_y_id = self._test_x[index_column_label]
        # drops the first column of the test set as the id so that it isnt included in the model
        self._test_x = self._test_x.drop(self._test_x.columns[0], axis=1)
        return None





