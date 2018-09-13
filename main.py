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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import sklearn




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

    model_df.switch_na_to_median_other_attribute('LotFrontage', 'Neighborhood')

    model_df.scatter_plot('SalePrice', 'LotFrontage')
    model_df.drop_outliers_target_greater_y_attribute_greater_x('SalePrice', 200000, 'LotFrontage', 300)
    model_df.scatter_plot('SalePrice', 'LotFrontage')

    model_df.switch_na_to_mode('MSZoning')
    model_df.drop_attribute_train_and_test('Utilities')#all features in train are all pub and 2 na, 'NoSewa' is in test set hence the attribute doesnt help in any way with the model so it is dropped

    model_df.boxplot('SalePrice', 'BsmtFullBath')

    model_df.switch_na_to_mode('MSZoning')
    model_df.switch_na_to_mode('Functional')#  in description says na should be functional which happens to be the mode
    model_df.switch_na_to_zero('BsmtHalfBath')#  assume na means no half bathroom
    model_df.switch_na_to_zero('BsmtFullBath')  # assume na means no full bathroom

    model_df.switch_na_to_mode('SaleType')

    model_df.switch_na_to_zero('GarageArea')
    model_df.switch_na_to_zero('GarageCars')
    model_df.switch_na_to_zero('TotalBsmtSF')
    model_df.switch_na_to_zero('BsmtUnfSF')
    model_df.switch_na_to_zero('BsmtFinSF2')
    model_df.switch_na_to_zero('BsmtFinSF1')
    model_df.switch_na_to_mode('KitchenQual')

    model_df.switch_na_to_mode('Exterior2nd')
    model_df.switch_na_to_mode('Exterior1st')

    #model_df.test_missing_data_ratio_and_bar_graph()  # print and show graph of the attributes with the largest amount of missing data
    #model_df.train_missing_data_ratio_and_bar_graph()

    model_df.index_column_drop_and_move_to_pred_Y('Id')  # drops id column from train_X and test_X to move it to _test_Y_id

    #model_df.boxcox_trans('SalePrice', 0)
    model_df.move_target_to_train_y('SalePrice')  # moves saleprice to train_Y

    print(model_df._train_Y)
    model_df._train_Y = boxcox(model_df._train_Y, 0.1)
    print(model_df._train_Y)


    model_df.convert_attribute_to_categorical('MSSubClass')
    model_df.convert_attribute_to_categorical('OverallCond')
    model_df.convert_attribute_to_categorical('YrSold')
    model_df.convert_attribute_to_categorical('MoSold')



    model_df.dim_data()


    ###################################################################################################################################################
    #now at a point where all the missing values are inputted!!!!
    ###################################################################################################################################################


    model_df.split_attributes()  # splits the attributes into a string dataset and a float + int dataset so that one hot encoding can be used
    model_df.one_hot_encoding()  # method to convert all the string attributes into one hot encoded

    model_df.boxcox_attributes(0.1) #must box cox transform the attributes first dont put in 0
    model_df.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y

    model_df._train_X_string.to_csv('Data_Out/_train_X_string_up.csv', index=False)
    model_df._test_X_string.to_csv('Data_Out/_test_X_string_up.csv', index=False)

    model_df._train_X_int_float.to_csv('Data_Out/_train_X_int_float.csv', index=False)
    model_df._test_X_int_float.to_csv('Data_Out/_test_X_int_float.csv', index=False)

    model_df.combine_string_int_float()  # combines the two objexts for both test_X and train_X


    model_df.dim_data()
    model_df.export_CSV_processed()  # exports the train_X, train_Y and test_X to a csv file





    a = model_df.lasso(0.01, 'SalePrice')
    print(a)

    a.to_csv('Data_Out/box_cox_target_lamda_0.01_alpha_0.1.csv', index=False)
    model_df.lasso_compare_alpha([1, 0.1, 0.01]).to_csv('Data_Out/Lasso_model_alpha_1_0point1_0point01.csv', index=False)# Run the function called, Lasso


    model_df.linear('SalePrice').to_csv('Data_Out/Linear_Model.csv', index=False) #  fixed


if __name__ == "__main__":
    main()
