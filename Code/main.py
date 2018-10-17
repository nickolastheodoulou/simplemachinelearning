import pandas as pd
from scipy.special import boxcox
from sklearn.metrics import confusion_matrix

from Code.Class_Data_Model import DataModel
from Code.Class_Data_Exploration import DataExploration


def main():
    ####################################################################################################################
    # Main used for house price data set

    #  at this point once the data has been explored, want train_Y to be in its own variable separate from train_X to
    #  pre-process the data train_X and test_X should not be combined at any point as the data should be preprocessed in
    #  one go for train_X but in a real world scenario, test_X may not come in as a large dataset
    model_house = DataModel(pd.read_csv("Data_In/House_Prices/train.csv"), pd.read_csv("Data_In/House_Prices/test.csv"))

    model_house.box_plot('SalePrice', 'YearBuilt')
    model_house.bar_graph_percentage('YrSold')
    model_house.line_graph_percentage_difference('YrSold')

    # dropped after looking at a scatter plot of the two attributes
    model_house.drop_outliers_target_less_y_attribute_greater_x('SalePrice', 300000, 'GrLivArea', 4000)
    model_house.drop_outliers_target_greater_y_attribute_greater_x('SalePrice', 200000, 'LotFrontage', 300)

    print(model_house._train_x.shape)
    # model_house.missing_data_ratio_and_bar_graph() used to decide how to handle missing values
    #  boxplot used to look at classified attributes
    model_house.switch_na_to_median_other_attribute('LotFrontage', 'Neighborhood')
    # all features in train are all pub and 2 na, 'NoSewa' is in test set hence the attribute doesnt help in any way
    # with the model so it is dropped
    model_house.drop_attribute_train_and_test('Utilities')
    attributes_to_none =["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", "GarageQual",
                         "GarageFinish", "GarageYrBlt", "GarageType", "BsmtFinType2", "BsmtExposure", "BsmtFinType1",
                         "BsmtCond", "BsmtQual", "MasVnrType"]

    attributes_to_zero = ['MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'GarageArea', 'GarageCars', 'TotalBsmtSF',
                          'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']

    attributes_to_mode = ['Electrical', 'MSZoning', 'Functional', 'SaleType', 'KitchenQual', 'Exterior2nd',
                          'Exterior1st']

    attributes_to_categorical = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']

    for x in attributes_to_none:
        model_house.switch_na_to_none(x)
    for x in attributes_to_zero:
        model_house.switch_na_to_zero(x)
    for x in attributes_to_mode:
        model_house.switch_na_to_mode(x)
    for x in attributes_to_categorical:
        model_house.convert_attribute_to_categorical(x)

    # drops id column from train_X and test_X to move it to _test_y_id
    model_house.index_column_drop_and_move_to_pred_y('Id')
    model_house.move_target_to_train_y('SalePrice')  # moves saleprice to train_Y
    model_house._train_y = boxcox(model_house._train_y, 0.1)
    ####################################################################################################################
    # all the missing values are inputted!!!!
    ####################################################################################################################

    # splits the attributes into a string dataset and a float + int dataset so that one hot encoding can be used
    model_house.split_attributes()
    # method to convert all the string attributes into one hot encoded
    model_house.one_hot_encoding()

    model_house.boxcox_attributes(0.1)  # must box cox transform the attributes first dont put in 0
    model_house.normalise_data()  # normalises train_X_int_float, test_X_int_float, train_Y
    model_house.combine_string_int_float()  # combines the two objects for both test_X and train_X

    variable_lasso_model = model_house.lasso(0.01, 'SalePrice')
    print(variable_lasso_model)

    variable_lasso_model.to_csv('Data_Out/box_cox_target_lamda_0.01_alpha_0.1.csv', index=False)
    model_house.lasso_compare_alpha([1, 0.1, 0.01]).to_csv('Data_Out/Lasso_model_alpha_1_0point1_0point01.csv',
                                                           index=False)  # Run the function called, Lasso

    model_house.linear('SalePrice').to_csv('Data_Out/Linear_Model.csv', index=False)  # functions correctly
    model_house.export_csv_processed()  # exports the train_X, train_Y and test_X to a csv file

    ####################################################################################################################

    ####################################################################################################################
    # Main used for iris data set
    iris_data = pd.read_csv('Data_In/Iris/iris.txt', header=None)
    iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'classification']

    model_iris = DataModel(iris_data.iloc[:70, :], iris_data.iloc[70:, :])

    model_iris.heatmap_correlated_attributes(5, 'classification')
    # print(model_iris._train_x.head())
    # print(model_iris._test_x.head())
    # model_iris.describe_attribute('sepal_length')
    # model_iris.histogram_and_q_q('sepal_length')

    model_iris.move_target_to_train_y('classification')
    model_iris.move_target_to_test_y('classification')

    print(confusion_matrix(model_iris._test_y, model_iris.svm()))
    print(confusion_matrix(model_iris._test_y, model_iris.neuralnetwork()))

    ####################################################################################################################

    ####################################################################################################################
    # Main used for DfBL data set

    model_DfBL = DataExploration(pd.read_csv("Data_In/DfBL/VehicleData_csv.csv"), 0)  # loads in the data

    print(model_DfBL._train_x.head())  # prints the first 5 columns of the data-set
    print(model_DfBL._train_x.shape)  # prints the dimension of the data set
    model_DfBL.missing_data_ratio_print()  # prints the percentage of missing values in the data set (NONE FOUND!)

    #  calls a function to combine the column VehicleType and Manufacturer to a new column named
    #  ManufacturerAndVehicleType
    model_DfBL.combine_columns("ManufacturerAndVehicleType", "VehicleType", "Manufacturer")

    print(model_DfBL._train_x.head())  # print the data_set to check it worked correctly

    #  model_DfBL._data_set.to_csv("Data_Out/data_set_out.csv")  # saves the pandas data frame to a CSV file

    model_DfBL.box_plot("ConditionScore", "FinancialYear")  # box plot of the Condition score for each Year
    model_DfBL.box_plot("ConditionScore", "Manufacturer")  # box plot of the Condition score for each Manufacturer

    model_DfBL.box_plot("ConditionScore", "ManufacturerAndVehicleType")  # box plot of the Condition score for each Year

    # function that prints the number of inspections each financial year.
    model_DfBL.column_value_count("FinancialYear")

    #  function that plots ans saves the percentage of what year each inspection occurs.
    model_DfBL.bar_graph_percentage("FinancialYear")

    #  function that plots the percentage difference of what year each inspection occurs.
    model_DfBL.line_graph_percentage_difference("FinancialYear")

    model_DfBL.triple_stacked_bar_graph('FinancialYear', 'VehicleType')

    ####################################################################################################################


if __name__ == "__main__":
    main()
