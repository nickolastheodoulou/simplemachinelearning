import pandas as pd # Load the Pandas libraries with alias 'pd'
import numpy as np
import matplotlib.pyplot as plt
from function_sale_price_against_attribute_scatter_plot import sale_price_against_attribute_scatter_plot


def main():
    train = pd.read_csv("Data/train.csv")  # read in train data from the csv file
    test = pd.read_csv("Data/test.csv") # read in test data from the csv file

    #print(train.head())#prints the first 5 columns of train
    # np.savetxt('Test/First_Floor_Sq_Foot.out', train['1stFlrSF'].values, delimiter=',')  # creates a file so that the data is easier to look at

    sale_price_against_attribute_scatter_plot('1stFlrSF')#function that takes in a column name of train as an attribute and creates a scatter plot of
    #the sale price against the attribute

if __name__ == "__main__":
    main()