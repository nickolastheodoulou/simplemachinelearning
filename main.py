import pandas as pd # Load the Pandas libraries with alias 'pd'
import numpy as np
import matplotlib.pyplot as plt
from Class_Analyse_Data import Analysing_Data
from Class_Data_Loader import data_matrix

def main():
    matrix = Analysing_Data()
    matrix.sale_price_against_attribute_scatter_plot('1stFlrSF')




    # print(train.head())#prints the first 5 columns of train
    # np.savetxt('Test/First_Floor_Sq_Foot.out', train['1stFlrSF'].values, delimiter=',')  # creates a file so that the data is easier to look at

    #sale_price_against_attribute_scatter_plot('1stFlrSF')  # function that takes in a column name of train as an attribute and creates a scatter plot of
    # the sale price against the attribute

if __name__ == "__main__":
    main()