import pandas as pd
from Class_Data_Model import data_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import boxcox1p, inv_boxcox


def main():
    matrix = data_model(pd.read_csv("Data_In/train.csv"), pd.read_csv("Data_In/test.csv"), 0, 0)  # load in the data, the other variables within the object will then be initialised later on using other functions
    matrix.dim_data()  # method that updates the dimension of the train and test data which is the 4th and 5th variable in object matrix
    #print(matrix._train_X.head(5))
    matrix.index_column_drop('Id')  # drops the first column of both test_X and train_X
    #matrix.dim_data()  # called again so that the dimension can be updated so the function that initialised train.Y with the correct values works properly

    #matrix.sale_price_against_attribute_scatter_plot('SalePrice', '1stFlrSF')  # creates the plot of sale price against house
    #matrix.describe_attribute('SalePrice')
    matrix.histogram_and_q_q('SalePrice')

    matrix.boxcox('SalePrice', -0.3)
    matrix.histogram_and_q_q('SalePrice')
    matrix.boxcox_inv('SalePrice', -0.3)

    #matrix.boxplot('Exterior1st', 'SalePrice')
    #matrix.heatmap()
    #matrix.heatmap_correlated_attributes(10, 'SalePrice')



'''
    # matrix._train_X.to_csv('Data_In/train_X.csv', index=False)
    #matrix.add_train_Y_to_train_X()
    matrix.move_target_to_train_y('SalePrice')  # moves the final column of train_X to train_Y
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
    matrix.lasso_compare_alpha([800, 900, 1000, 1100, 1200]).to_csv('Data_Out/Lasso_model_alpha_800_900_1000_1100_1200.csv', index=False)# Run the function called, Lasso

    matrix.lasso(1000).to_csv('Data_Out/Lasso_model_alpha_1000_pipeline.csv', index=False)


    # The coefficients
    # print('Coefficients: \n', regr.coef_)
'''

if __name__ == "__main__":
    main()
