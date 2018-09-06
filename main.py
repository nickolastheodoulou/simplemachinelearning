import pandas as pd # Load the Pandas libraries with alias 'pd'
import numpy as np
import matplotlib.pyplot as plt
from Class_Data_Exploration import Data_Exploration
from Class_Data_Loader import data_matrix
from collections import Counter#used to count size of each classification for an attribute


def main():
    matrix = Data_Exploration(pd.read_csv("Data/test.csv"), pd.read_csv("Data/train.csv"))
    #matrix = data_matrix()#load in the data, the other variables within the object will then be
    # initialised later on using other functions
    matrix.dim_data()#method that updates the dimension of the train and test data which is the 4th and 5th variable in object matrix

    print("The dimension of train_X is: ", matrix._train_X_dim, "The dimension of test_X is: ", matrix._test_X_dim)#prints dimensions of train and test

    print(matrix._train_X.head())#prints the first 5 rows of train_X
    print(matrix._test_X.head())
    print("The dimension of train_X is: ", matrix._train_X_dim, "The dimension of test_X is: ", matrix._test_X_dim)

    #as can be seen, both test_X and train_X have the id within the first column, this will need to be dropped.
    #test_X has an extra column: SalePrice which needs to be moved over to train.Y

    matrix.first_column_drop()#drops the first column of both test_X and train_X
    matrix.dim_data()#called again so that the dimension can be updated so the function that initialised train.Y with the correct values works properly
    matrix.move_classification_to_train_y()#moves the final column of train_X to train_Y
    matrix.dim_data()  # called again to verify everything worked correctly with the following print statement
    print("The dimension of train_X is: ", matrix._train_X_dim, "The dimension of test_X is: ", matrix._test_X_dim)

    print(matrix._train_Y.head())
    print(matrix._train_X.head())

    matrix.sale_price_against_attribute_scatter_plot('1stFlrSF')#creates the plot of sale price against house

    matrix.split_attributes()

    print(matrix._train_X_int_float.head())
#need to come up with one method that does both by passing in either matrix.test or matrix._train

    '''


    one_hot_encoded_train = pd.get_dummies(string_train)#method to convert all the string attributes into one hot encoded
    print(one_hot_encoded_train.head())#print one_hot encoded to ensure it actually works

    np.savetxt('Test/test.out', one_hot_encoded_train, delimiter=',')  # creates a file so that the data is easier to look at


'''



if __name__ == "__main__":
    main()
