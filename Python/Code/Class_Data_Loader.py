import pandas as pd


class DataLoader:  # class that loads in the data
    def __init__(self, dataset):  # initialise the object with the test and train matrices from the CSV file

        self._dataset = dataset  # dataframe
        self._target = 0  # target attribute
        self._id = 0  # id of each column
        # the underscore means that the members are protected

        print(self, 'created')  # print statement to show that the object is created

    def __del__(self):  # destroy object with a print statement
        print(self, 'destroyed')

    def dim_data(self):  # function that prints the dimensions of the dataset
        print('The dimensions of the dataset is', self._dataset.shape)
