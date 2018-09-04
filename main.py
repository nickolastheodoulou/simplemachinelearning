import pandas as pd # Load the Pandas libraries with alias 'pd'
import numpy as np
import matplotlib.pyplot as plt
from Class_Data_Exploration import Data_Exploration
from Class_Data_Loader import data_matrix
from collections import Counter#used to count size of each classification for an attribute

def main():
    matrix = Data_Exploration()#create the object matrix that has variables test and train matrix
    matrix.sale_price_against_attribute_scatter_plot('1stFlrSF')#create a scatterplot of sale price against first floor square foot
    matrix.sale_price_against_attribute_scatter_plot('2ndFlrSF')
    matrix.sale_price_against_attribute_scatter_plot('BsmtFinSF1')
    matrix.sale_price_against_attribute_scatter_plot('BsmtFinSF2')
    matrix.sale_price_against_attribute_scatter_plot('TotalBsmtSF')
    matrix.sale_price_against_attribute_scatter_plot('MSSubClass')

    print(Counter(matrix._train["ExterQual"]))#prints the number of each classification for ExterQual


    # print(train.head())#prints the first 5 columns of train
    # np.savetxt('Test/First_Floor_Sq_Foot.out', train['1stFlrSF'].values, delimiter=',')  # creates a file so that the data is easier to look at

    #sale_price_against_attribute_scatter_plot('1stFlrSF')  # function that takes in a column name of train as an attribute and creates a scatter plot of
    # the sale price against the attribute

if __name__ == "__main__":
    main()