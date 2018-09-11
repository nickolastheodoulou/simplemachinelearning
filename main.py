import pandas as pd
from Class_Data_Model import DataModel
from Class_Data_Exploration import DataExploration
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import boxcox1p, inv_boxcox1p


def main():

    # load in the data to the exploration class so that the pre-processing isn't interfered with when
    #  the train and testset are modified such as when id is dropped or train_Y is moved
    exploration_df = DataExploration(pd.read_csv("Data_In/train.csv"), pd.read_csv("Data_In/test.csv"))
    #exploration_df.dim_data()  # method that updates the dimension of the train and test data which is the 4th and 5th variable in object exploration_df

    #print(exploration_df._train_X.head(5))
    exploration_df.index_column_drop_and_move_to_pred_Y('Id')  # drops the first column of both test_X and train_X

    # creates the plot of sale price against house
    #exploration_df.sale_price_against_attribute_scatter_plot('SalePrice', '1stFlrSF')
    #exploration_df.describe_attribute('SalePrice')
    #exploration_df.histogram_and_q_q('SalePrice')

    #exploration_df.boxcox('SalePrice', -0.3)
    #exploration_df.histogram_and_q_q('SalePrice')
    #exploration_df.boxcox_inv('SalePrice', -0.3)

    #exploration_df.boxplot('Exterior1st', 'SalePrice')
    #exploration_df.heatmap()
    #exploration_df.heatmap_correlated_attributes(10, 'SalePrice')
    #exploration_df.missing_data_ratio_and_bar_graph('SalePrice')

    #  at this point once the data has been explored, want train_Y to be in its own variable separate from train_X to
    #  pre-process the data train_X and test_X should not be combined at any point as the data should be preprocessed in
    #  one go for train_X but in a real world scenario, test_X may not come in as a large dataset
    model_df = DataModel(pd.read_csv("Data_In/train.csv"), pd.read_csv("Data_In/test.csv"))
    model_df.move_target_to_train_y('SalePrice')
    model_df.index_column_drop_and_move_to_pred_Y('Id')
    #print(model_df._train_Y)
    #model_df.boxcox_target(0)
    print(model_df._train_Y)
    model_df.dim_data()

    model_df.split_attributes()  # splits the attributes into a string dataset and a float + int dataset so that one hot encoding can be used
    model_df.one_hot_encoding()  # method to convert all the string attributes into one hot encoded
    model_df._train_X_string.to_csv('Data_Out/_train_X_string.csv', index=False)
    model_df._test_X_string.to_csv('Data_Out/_test_X_string.csv', index=False)

    # print(matrix._train_X_string.head())#print one_hot encoded to ensure it actually works

    model_df.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y
    model_df.fill_missing_values()  # fills in the missing values of train_X_int_float
    model_df.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y

    model_df.combine_string_int_float()  # combines the two objexts for both test_X and train_X
    model_df.export_CSV_processed()  # exports the train_X, train_Y and test_X to a csv file


    #model_df.lasso_compare_alpha([800, 900, 1000, 1100, 1200]).to_csv('Data_Out/Lasso_model_alpha_800_900_1000_1100_1200.csv', index=False)# Run the function called, Lasso

    #model_df.lasso(1000).to_csv('Data_Out/Lasso_model_alpha_1000_pipeline.csv', index=False)

    #print(model_df.lasso(1000))

    #attribute = 'SalePrice'
    #lamda = 0
    #model_df._pred_Y[attribute] = inv_boxcox1p(model_df._pred_Y[attribute], lamda)
    #print(model_df.lasso(1000))

    regr = sklearn.linear_model.LinearRegression()  # Create linear regression object
    regr.fit(model_df._train_X, model_df._train_Y)  # Train the model using the training sets
    pred_Y_model = regr.predict(model_df._test_X)  # Make predictions using the testing set
    pred_Y_model = pd.DataFrame(data=pred_Y_model, columns={'Target'})  #
    pred_Y_model = pd.concat([model_df._test_Y_id, pred_Y_model], axis=1)

    print(pred_Y_model)

    print(model_df.linear())
    #model_df.linear().to_csv('Data_Out/linear_model.csv', index=False)  # run the linear model and save output to a CSV file
    #print(model_df.lasso(1000))




if __name__ == "__main__":
    main()
