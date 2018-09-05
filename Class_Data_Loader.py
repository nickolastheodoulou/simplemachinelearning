import pandas as pd
import matplotlib.pyplot as plt

class data_matrix:  # class that creates a test and train matrix so that they can be inherited into other classes for functions rather than being constantly
    # created each time a function is called
    def __init__(self):
        self._test = pd.read_csv("Data/test.csv")  # read in the test data from the csv file
        self._train = pd.read_csv("Data/train.csv")  # read in train data from the csv file
        # the underscore means that the members are protected

    def id_drop(self):  # method that plots sales against an attribute
        self._train = self._train.drop(self._train.columns[0], axis=1)#drops the first column of the train set as the id so that it isnt included in the model
        self._test = self._test.drop(self._test.columns[0], axis=1)  # drops the first column of the test set as the id so that it isnt included in the model

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