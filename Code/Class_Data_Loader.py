import pandas as pd


class DataLoader:  # class that creates the data matrix by initializing test_X and train_X from main
    def __init__(self, train_X, test_X):#initialise the object with the test and train matrices from the CSV file

        self._test_X = test_X                           #test dataframe
        self._train_X = train_X                         #train dataframe
        self._train_Y = 0                         #train_Y and test_Y given as arguments as in some instances the data will already be split
        self._test_Y = 0
        self._test_Y_id = 0                                #id column later used to add to Pred_Y if the index_column is dropped
        self.train_X_and_test_X = 0
        # the underscore means that the members are protected
        print(self, 'created')

    def __del__(self):
        print(self, 'destroyed')

    def dim_data(self):  # function that finds the dimensions of both the train and test set and stores them in the object
        print('The dimensions of train_X is', self._train_X.shape, 'and the dimension of test_X is ', self._test_X.shape)

    def add_train_Y_to_train_X(self):#function to add train_Y to train_X if it is separate so that only 1 function is made to analyse the data in train_X,
        #  there is later a function that splits them in the data_preprocessing class
        self._train_X = pd.concat([self._train_X, self._train_Y], axis=1)  # combines the train_X

    def move_target_to_train_y(self, target):  # function that removes the last column of Train_X and puts it into Train_Y (only called if an index column is in the datasets
        final_column = self._train_X.columns.get_loc(target)  # finds the target column by name passed through from the function
        d = {target: self._train_X.iloc[:, final_column]}  # updates the dataframe train_Y using the index column
        self._train_Y = pd.DataFrame(data=d)
        self._train_X = self._train_X.drop(self._train_X.columns[final_column], axis=1)  # drops the first column of the train set as it has been moved
        return None

    def move_target_to_test_y(self, target):  # function that removes the last column of Test_X and puts it into Test_Y
        final_column = self._test_X.columns.get_loc(target)  # finds the target column by name
        d = {target: self._test_X.iloc[:, final_column]}  # updates the dataframe train_Y using the index column
        self._test_Y = pd.DataFrame(data=d)# updates the dataframe train_Y
        self._test_X = self._test_X.drop(self._test_X.columns[final_column], axis=1)  # drops the first column of the train set as it has been moved
        return None

    def index_column_drop_and_move_to_pred_Y(self, index_column_label):  # method that drops the first column of both train_X and test_X
        self._train_X = self._train_X.drop(self._train_X.columns[0], axis=1)  # drops the first column of the train set as the id so that it isnt included in the model
        self._test_Y_id = self._test_X[index_column_label]  # define id so that it can be added to pred_Y
        self._test_X = self._test_X.drop(self._test_X.columns[0], axis=1)  # drops the first column of the test set as the id so that it isnt included in the model
        return None





