import pandas as pd
from sklearn.model_selection import KFold
from sklearn import neighbors
import numpy as np

from Class_Data_Modeler import DataModeler


def main():
    car_insurance_model = DataModeler(pd.read_csv("Data_In/DS_Assessment.csv"))  # first load the data
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)
    print(car_insurance_model._data_set.head())  # prints the first 5 columns of the data set

    # displays and saves a bar graph showing the percentage of each value for the column Age in the data set
    car_insurance_model.bar_graph_attribute('Age')
    car_insurance_model.bar_graph_attribute_by_classification('Age')
    car_insurance_model.describe_attribute('Age')  # prints a summary of the distribution of the column 'Age'
    car_insurance_model.histogram_and_q_q('Age')



    # displays and saves a bargraph of the percentage of missing values
    car_insurance_model.missing_data_ratio_bar_graph()



    car_insurance_model.one_hot_encode_attribute('Marital_Status')

    car_insurance_model.add_day_of_week_attribute()
    car_insurance_model.one_hot_encode_attribute('days_of_the_week')
    car_insurance_model.drop_attribute('Date')

    # car_insurance_model.scatter_plot_by_classification("Tax", "Price")

    # found that tax and price follow two linear equations using car_insurance_model.scatter_plot("Tax", "Price")
    # the cutoff between following either equation was when the tax was between a value of 32 to 35:
    # typically when tax < 34, tax = 0.05 * price and when tax > 34, tax = 0.1 * price
    # hence this can be used to impute missing values more accurately

    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)


    car_insurance_model.normalise_attribute('Veh_Mileage')
    car_insurance_model.normalise_attribute('Credit_Score')
    car_insurance_model.normalise_attribute('License_Length')
    car_insurance_model.normalise_attribute('Veh_Value')
    car_insurance_model.normalise_attribute('Price')
    car_insurance_model.normalise_attribute('Age')
    car_insurance_model.normalise_attribute('Tax')

    car_insurance_model.impute_price()
    # set the final few values to the mean
    car_insurance_model.impute_mean('Price')

    car_insurance_model.impute_tax()
    # set the final few values to the mean
    car_insurance_model.impute_mean('Tax')

    car_insurance_model.impute_mode('Veh_Mileage')
    # car_insurance_model.new_column_infinite_credit_score()
    car_insurance_model.impute_median('Credit_Score')

    # should try to impute by first categorising by Maritial_Status
    car_insurance_model.impute_median('License_Length')

    car_insurance_model.impute_mode('Veh_Value')  # should find a better way

    car_insurance_model._data_set['Age'] = car_insurance_model._data_set['Age'].fillna(37)  # should find a better way
    # car_insurance_model.missing_data_ratio_bar_graph()  # prints the number of missing values in each column

    car_insurance_model.attribute_value_count('Age')  # counts the number of different values in the attribute
    car_insurance_model.attribute_value_count_by_classification('Age')
    car_insurance_model.bar_graph_attribute_by_classification('Age')

    # car_insurance_model.impute_knn(3) # imputing using knn from fancyimpute package found to be to inefficient

    car_insurance_model.drop_all_na()
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)

    # car_insurance_model.bar_graph_attribute('Veh_Value')
    # car_insurance_model.bar_graph_attribute_by_classification('Tax')
    # car_insurance_model.scatter_plot_by_classification('Price', 'Veh_Mileage')

    # car_insurance_model.scatter_plot('Age', 'Tax')

    # car_insurance_model.histogram_and_q_q('Price')

    # max_price = car_insurance_model._data_set['Price'].max()
    # car_insurance_model._data_set['Price'] = max_price + 1 - car_insurance_model._data_set['Price']
    # car_insurance_model.boxcox_trans_attribute('Price', 0.1)
    # car_insurance_model._data_set['Price'] = np.sqrt(car_insurance_model._data_set['Price'])
    # car_insurance_model.histogram_and_q_q('Price')

    # car_insurance_model._data_set['Credit_Score'] = car_insurance_model._data_set['Credit_Score'].replace(9999, 999)

    '''
    car_insurance_model.boxcox_trans_attribute('Price', 0.1)
    max_price = car_insurance_model._data_set['Price'].max()
    car_insurance_model._data_set['Price'] = max_price + 1 - car_insurance_model._data_set['Price']
    '''


    # car_insurance_model.histogram_and_q_q('Price')

    # car_insurance_model.missing_data_ratio_print()  # prints the number of missing values in each column

    # car_insurance_model.drop_all_na()
    # print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)

    car_insurance_model.shuffle_data_set()  # shuffle the data set before splitting
    #   must split data before fitting model
    car_insurance_model.split_data_set_into_train_x_test_x_train_y_test_y('Sale', 0.25, 2)

    # car_insurance_model.knn_model(5, 10)
    # car_insurance_model.svm_model('auto', 10)


if __name__ == "__main__":
    main()
