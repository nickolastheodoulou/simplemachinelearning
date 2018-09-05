import pandas as pd # Load the Pandas libraries with alias 'pd'
import numpy as np
import matplotlib.pyplot as plt
from Class_Data_Exploration import Data_Exploration
from Class_Data_Loader import data_matrix
from collections import Counter#used to count size of each classification for an attribute


def main():
    matrix = Data_Exploration()#create the object matrix that has variables test and train matrix
    #matrix.sale_price_against_attribute_scatter_plot('1stFlrSF')#create a scatterplot of sale price against first floor square foot

    matrix.id_drop()#method that drops the id of both the test and train dataset by removing the first column



#need to come up with one method that does both by passing in either matrix.test or matrix._train
    string_train = matrix.split_string_attributes_train()#splits train into the string attributes
    int_float_train = matrix.split_int_float_attributes_train()#splits train into int and float attributes

    string_test = matrix.split_string_attributes_test()#split test to make sure functions in class work correctly
    int_float_test = matrix.split_int_float_attributes_test()


    print("The size of the whole train is", matrix._train.shape)
    print("The size of the string train is", string_train.shape)
    print("The size of the int_float train is", int_float_train.shape)

    print("The size of the whole test is", matrix._test.shape)
    print("The size of the string test is", string_test.shape)
    print("The size of the int_float test is", int_float_test.shape)#print functions to check all the numbers add correctly


    one_hot_encoded_train = pd.get_dummies(string_train)#method to convert all the string attributes into one hot encoded
    print(one_hot_encoded_train.head())#print one_hot encoded to ensure it actually works

    np.savetxt('Test/test.out', one_hot_encoded_train, delimiter=',')  # creates a file so that the data is easier to look at






if __name__ == "__main__":
    main()
