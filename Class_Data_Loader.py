import pandas as pd
import matplotlib.pyplot as plt

class data_matrix:  # class that creates the data matrix by initializing test_X and train_X from main
    def __init__(self, test_X, train_X):
        self._test_X = test_X
        self._train_X = train_X
        self._train_Y = 0
        self._test_X_dim = 0
        self._train_X_dim = 0
        # the underscore means that the members are protected

    def dim_data(self):#function that finds the dimensions of both the train and test set and stores them in the object
        self._train_X_dim = self._train_X.shape
        self._test_X_dim = self._test_X.shape

    def first_column_drop(self):  # method that drops the first column of both train_X and test_X
        self._train_X = self._train_X.drop(self._train_X.columns[0], axis=1)#drops the first column of the train set as the id so that it isnt included in the model
        self._test_X = self._test_X.drop(self._test_X.columns[0], axis=1)  # drops the first column of the test set as the id so that it isnt included in the model

    def move_classification_to_train_y(self):
        final_column = self._train_X_dim[1] - 1
        self._train_Y = self._train_X.iloc[:, final_column]
        self._train_X = self._train_X.drop(self._train_X.columns[(self._train_X.shape[1]-1)], axis=1)  # drops the first column of the train set as the id so that it isnt included in the model



'''
    def split_string_attributes_train(self):#method that returns the float attributes within the train dataset
        self.matrix = self._train.select_dtypes(include=['object']).copy()#creates protected object _string_train that contains all the "object" datatypes in train
        return self.matrix

    def split_string_attributes_test(self):#method that returns the float attributes within the test dataset
        self.matrix = self._test.select_dtypes(include=['object']).copy()#creates protected object _string_test that contains all the "object" datatypes in train
        return self.matrix

    def split_int_float_attributes_train(self):#method that returns the int and float attributes within the train dataset
        self.matrix = self._train.select_dtypes(include=['int64', 'float64']).copy()
        return self.matrix

    def split_int_float_attributes_test(self):#method that returns the int and float attributes within the test dataset
        self.matrix = self._test.select_dtypes(include=['int64', 'float64']).copy()
        return self.matrix

'''