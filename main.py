import pandas as pd # Load the Pandas libraries with alias 'pd'
import numpy as np
import matplotlib.pyplot as plt
from Class_Data_Exploration import Data_Exploration
from Class_Data_Loader import data_matrix
from collections import Counter#used to count size of each classification for an attribute


def main():
    matrix = Data_Exploration()#create the object matrix that has variables test and train matrix
    #matrix.sale_price_against_attribute_scatter_plot('1stFlrSF')#create a scatterplot of sale price against first floor square foot



    objects_train = matrix._train.select_dtypes(include=['object']).copy()
    #print(objects_train.head())

    one_hot_train =pd.get_dummies(objects_train)
    #print(a)
    np.savetxt('Test/one_hot_train.out', one_hot_train, delimiter=',', fmt='%i')  # creates a file so that the data is easier to look at

if __name__ == "__main__":
    main()
