import pandas as pd # Load the Pandas libraries with alias 'pd'
from Class_Data_Model import data_model
import sklearn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def main():
    matrix = data_model(pd.read_csv("Data_In/train.csv"), pd.read_csv("Data_In/test.csv"), 0, 0)  # load in the data, the other variables within the object will then be initialised later on using other functions
    matrix.dim_data()  # method that updates the dimension of the train and test data which is the 4th and 5th variable in object matrix
    matrix.index_column_drop('Id')  # drops the first column of both test_X and train_X
    matrix.dim_data()  # called again so that the dimension can be updated so the function that initialised train.Y with the correct values works properly
    matrix.sale_price_against_attribute_scatter_plot('SalePrice', '1stFlrSF')  # creates the plot of sale price against house

    matrix.describe_attribute('1stFlrSF')
    matrix.describe_attribute('SalePrice')
    matrix.histogram('1stFlrSF')
    matrix.histogram('SalePrice')


    # matrix._train_X.to_csv('Data_In/train_X.csv', index=False)



    #matrix.add_train_Y_to_train_X()
    #matrix.move_target_to_train_y('SalePrice')  # moves the final column of train_X to train_Y


'''
    matrix.dim_data()
    matrix.split_attributes()  # splits the attributes into a string dataset and a float + int dataset so that one hot encoding can be used
    matrix.one_hot_encoding()  # method to convert all the string attributes into one hot encoded
    matrix._train_X_string.to_csv('Data_Out/_train_X_string.csv', index=False)
    matrix._test_X_string.to_csv('Data_Out/_test_X_string.csv', index=False)

    # print(matrix._train_X_string.head())#print one_hot encoded to ensure it actually works

    matrix.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y
    matrix.fill_missing_values()  # fills in the missing values of train_X_int_float
    matrix.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y

    matrix.combine_string_int_float()  # combines the two objexts for both test_X and train_X
    matrix.export_CSV_processed()  # exports the train_X, train_Y and test_X to a csv file

    matrix.linear().to_csv('Data_Out/linear_model.csv', index=False)  # run the linear model and save output to a CSV file
    matrix.lasso([1000, 2000, 3000, 4000, 5000]).to_csv('Data_Out/Lasso_model_alpha_1000_2000_3000_4000_5000.csv', index=False)# Run the function called, Lasso

    regr = sklearn.linear_model.Lasso(alpha=1000)
    regr.fit(matrix._train_X, matrix._train_Y)
    Pred_Y_list = regr.predict(matrix._test_X)  # Make predictions using the testing set
    Pred_Y = pd.DataFrame(data=Pred_Y_list, columns={'SalePrice'})  #
    linear_model = pd.concat([matrix._pred_Y, Pred_Y], axis=1)
    linear_model.to_csv('Data_Out/Lasso_model_alpha_1000.csv', index=False)

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
'''

if __name__ == "__main__":
    main()
