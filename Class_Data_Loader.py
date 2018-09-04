import pandas as pd
import matplotlib.pyplot as plt

class data_matrix:  # class that creates a test and train matrix so that they can be inherited into other classes for functions rather than being constantly
    # created each time a function is callsed
    def __init__(self):
        self.test = pd.read_csv("Data/test.csv")  # read in the test data from the csv file
        self.train = pd.read_csv("Data/train.csv")  # read in train data from the csv file
        # the underscore means that the members are protected



        

