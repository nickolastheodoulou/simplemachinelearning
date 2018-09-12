import pandas as pd
from Class_Data_Model import DataModel
from Class_Data_Exploration import DataExploration
from sklearn import datasets, linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from Class_Data_Exploration import DataExploration
from scipy.special import boxcox, inv_boxcox, boxcox1p, inv_boxcox1p


def main():
    #  at this point once the data has been explored, want train_Y to be in its own variable separate from train_X to
    #  pre-process the data train_X and test_X should not be combined at any point as the data should be preprocessed in
    #  one go for train_X but in a real world scenario, test_X may not come in as a large dataset
    model_df = DataModel(pd.read_csv("Data_In/train.csv"), pd.read_csv("Data_In/test.csv"))

    model_df.scatter_plot('SalePrice', 'GrLivArea')  # creates the plot of sale price against house
    model_df.drop_outliers_target_less_y_attribute_greater_x('SalePrice', 300000, 'GrLivArea', 4000)
    model_df.scatter_plot('SalePrice', 'GrLivArea')  # creates the plot of sale price against house

    model_df.heatmap_correlated_attributes(10, 'SalePrice')

    #model_df.sale_price_against_attribute_scatter_plot('SalePrice', 'GarageCars')  # creates the plot of sale price against house



    model_df.switch_na_to_none("PoolQC")
    model_df.switch_na_to_none("MiscFeature")
    model_df.switch_na_to_none("Alley")
    model_df.switch_na_to_none("Fence")
    model_df.switch_na_to_none("FireplaceQu")

    model_df.switch_na_to_none("GarageCond")
    model_df.switch_na_to_none("GarageQual")
    model_df.switch_na_to_none("GarageFinish")
    model_df.switch_na_to_none("GarageYrBlt")
    model_df.switch_na_to_none("GarageType")

    model_df.switch_na_to_none("BsmtFinType2")
    model_df.switch_na_to_none("BsmtExposure")

    model_df.switch_na_to_none("BsmtFinType1")
    model_df.switch_na_to_none("BsmtCond")
    model_df.switch_na_to_none("BsmtQual")
    model_df.switch_na_to_none("MasVnrType")
    model_df.switch_na_to_zero('MasVnrArea')

    model_df.switch_na_to_mode('Electrical')

    # Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    model_df._train_X["LotFrontage"] = model_df._train_X.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


    model_df.scatter_plot('SalePrice', 'LotFrontage')
    model_df.drop_outliers_target_greater_y_attribute_greater_x('SalePrice', 200000, 'LotFrontage', 300)
    model_df.scatter_plot('SalePrice', 'LotFrontage')

    #model_df.missing_data_ratio_and_bar_graph()  # print and show graph of the attributes with the largest amount of missing data


    model_df.index_column_drop_and_move_to_pred_Y('Id')  # drops id column from train_X and test_X to move it to _test_Y_id

    model_df.move_target_to_train_y('SalePrice')  # moves saleprice to train_Y
    model_df.export_CSV_processed()  # exports the train_X, train_Y and test_X to a csv file
'''
    model_df.split_attributes()  # splits the attributes into a string dataset and a float + int dataset so that one hot encoding can be used
    model_df.one_hot_encoding()  # method to convert all the string attributes into one hot encoded
    model_df._train_X_string.to_csv('Data_Out/_train_X_string.csv', index=False)
    model_df._test_X_string.to_csv('Data_Out/_test_X_string.csv', index=False)

    # print(matrix._train_X_string.head())#print one_hot encoded to ensure it actually works

    model_df.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y
    model_df.fill_missing_values()  # fills in the missing values of train_X_int_float
    model_df.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y

    model_df._train_X_int_float.to_csv('Data_Out/train_X_int_float.csv', index=False)
    model_df._test_X_int_float.to_csv('Data_Out/test_X_int_float.csv', index=False)

    model_df.combine_string_int_float()  # combines the two objexts for both test_X and train_X
    model_df.export_CSV_processed()  # exports the train_X, train_Y and test_X to a csv file


    model_df.lasso_compare_alpha([800, 900, 1000]).to_csv('Data_Out/Lasso_model_alpha_800_900_1000.csv', index=False)# Run the function called, Lasso

    model_df.lasso(900, 'SalePrice').to_csv('Data_Out/Lasso_model_alpha_900_pipeline.csv', index=False)

    #model_df.linear().to_csv('Data_Out/Linear_Model.csv', index=False) #  still gives terrible predictions
'''

if __name__ == "__main__":
    main()
