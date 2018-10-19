import pandas as pd
import matplotlib.pyplot as plt

from Class_Data_Modeler import DataModeler


def main():
    car_insurance_model = DataModeler(pd.read_csv("Data_In/DS_Assessment.csv"))  # first load in the data

    # print the dimension of the data set
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)

    car_insurance_model.column_value_count('Age')  # counts the number of different values in the 'Age' column
    # displays and saves a bar graph showing the percentage of each value for the column Age in the data set
    # car_insurance_model.bar_graph_distribution('Age')

    car_insurance_model.describe_attribute('Age')  # prints a summary of the distribution of the column 'Age'
    car_insurance_model.missing_data_ratio_print()  # prints the number of missing values in each column

    # displays and saves a bargraph of the percentage of missing values
    # car_insurance_model.missing_data_ratio_bar_graph()

    print(car_insurance_model._data_set.head())  # prints the first 5 columns of the data set
    car_insurance_model.shuffle_data_set()  # shuffle the data set
    print(car_insurance_model._data_set.head())  # print again to check the data is shuffled

    car_insurance_model._data_set.to_csv('Data_Out/pre_one_hot.csv', index=False)
    car_insurance_model.one_hot_encode_attribute('Marital_Status')
    # car_insurance_model.drop_attribute('Marital_Status')
    car_insurance_model._data_set.to_csv('Data_Out/post_one_hot.csv', index=False)

    car_insurance_model.add_day_of_week_attribute()
    car_insurance_model.one_hot_encode_attribute('days_of_the_week')

    car_insurance_model.drop_attribute('Date')

    car_insurance_model.drop_all_na()

    car_insurance_model.scatter_plot('Age', 'Tax')

    car_insurance_model.scatter_plot_classification_colour('Age', 'Tax')




    # car_insurance_model.histogram_and_q_q('Price')

    # max_price = car_insurance_model._data_set['Price'].max()
    # car_insurance_model._data_set['Price'] = max_price + 1 - car_insurance_model._data_set['Price']
    # car_insurance_model.boxcox_trans_attribute('Price', 0.1)
    # car_insurance_model._data_set['Price'] = np.sqrt(car_insurance_model._data_set['Price'])
    # car_insurance_model.histogram_and_q_q('Price')

    # car_insurance_model._data_set['Credit_Score'] = car_insurance_model._data_set['Credit_Score'].replace(9999, 999)
    # car_insurance_model._data_set.to_csv('Data_Out/missing_values_dropped.csv', index=False)

    # car_insurance_model.boxcox_trans_attribute('Credit_Score', 0.1)
    # car_insurance_model.histogram_and_q_q('Credit_Score')

    '''
    car_insurance_model.boxcox_trans_attribute('Veh_Value', 0.1)
    car_insurance_model.boxcox_trans_attribute('Tax', 0.1)
    car_insurance_model.boxcox_trans_attribute('Price', 0.1)
    car_insurance_model.boxcox_trans_attribute('Credit_Score', 0.1)
    #car_insurance_model.boxcox_trans_attribute('License_Length', 0.1)
    '''

    car_insurance_model.normalise_attribute('Veh_Mileage')
    car_insurance_model.normalise_attribute('Credit_Score')
    car_insurance_model.normalise_attribute('License_Length')
    car_insurance_model.normalise_attribute('Veh_Value')
    car_insurance_model.normalise_attribute('Price')
    car_insurance_model.normalise_attribute('Age')
    car_insurance_model.normalise_attribute('Tax')

    car_insurance_model.histogram_and_q_q('Price')

    # car_insurance_model.missing_data_ratio_print()  # prints the number of missing values in each column
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)

    #  car_insurance_model._data_set.to_csv('Data_Out/missing_values_dropped.csv', index=False)

    #   must split data before fitting model
    car_insurance_model.split_data_set_into_train_x_test_x_train_y_test_y('Sale', 0.5, 0)

    # car_insurance_model.knn_model(5)
    # car_insurance_model.svm_model('auto')


if __name__ == "__main__":
    main()
