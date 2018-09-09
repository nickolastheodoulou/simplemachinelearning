class DataLoader:  # class that creates the data matrix by initializing test_X and train_X from main
    def __init__(self, train_X, test_X, train_Y, test_Y):#initialise the object with the test and train matrices from the CSV file

        self._test_X = test_X                           #test dataframe
        self._train_X = train_X                         #train dataframe
        self._train_Y = train_Y                         #train_Y and test_Y given as arguments as in some instances the data will already be split
        self._test_Y = test_Y
        self._pred_Y = 0                                #id column later used to add to Pred_Y if the index_column is dropped
        # the underscore means that the members are protected

    def dim_data(self):  # function that finds the dimensions of both the train and test set and stores them in the object
        print('The dimensions of train_X is', self._train_X.shape, 'and the dimension of test_X is ', self._test_X.shape)

    def index_column_drop(self, index_column_label):  # method that drops the first column of both train_X and test_X
        self._train_X = self._train_X.drop(self._train_X.columns[0], axis=1)  # drops the first column of the train set as the id so that it isnt included in the model
        self._pred_Y = self._test_X[index_column_label]  # define id so that it can be added to pred_Y
        self._test_X = self._test_X.drop(self._test_X.columns[0], axis=1)  # drops the first column of the test set as the id so that it isnt included in the model
        return None

    def move_target_to_train_y(self, target):  # function that removes the last column of Train_X and puts it into Train_Y (only called if an index column is in the datasets
        final_column = self._train_X.columns.get_loc(target)  # finds the target column by name passed through from the function
        self._train_Y = self._train_X.iloc[:, final_column]  # updates the dataframe train_Y using the index column
        self._train_X = self._train_X.drop(self._train_X.columns[final_column], axis=1)  # drops the first column of the train set as it has been moved
        return None

    def move_target_to_test_y(self, target):  # function that removes the last column of Train_X and puts it into Train_Y
        final_column = self._test_X.columns.get_loc(target)  # finds the target column by name
        self._test_Y = self._test_X.iloc[:, final_column]  # updates the dataframe train_Y
        self._test_X = self._test_X.drop(self._train_X.columns[final_column], axis=1)  # drops the first column of the train set as it has been moved
        return None



