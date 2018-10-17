import pandas as pd

from Code.Class_Data_Modeler import DataModeler


def main():
    car_insurance_model = DataModeler(pd.read_csv("Data_In/DS_Assessment.csv"))  # first load in the data

    # print the dimension of the data set
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)

    car_insurance_model.column_value_count('Age')  # counts the number of different values in the 'Age' column
    # displays and saves a bar graph showing the percentage of each value for the column Age in the data set
    car_insurance_model.bar_graph_distribution('Age')

    car_insurance_model.describe_attribute('Age')  # prints a summary of the distribution of the column 'Age'
    car_insurance_model.missing_data_ratio_print()  # prints the number of missing values in each column

    # displays and saves a bargraph of the percentage of missing values
    car_insurance_model.missing_data_ratio_bar_graph()

    print(car_insurance_model._data_set.head())  # prints the first 5 columns of the data set
    car_insurance_model.shuffle_data_set()  # shuffle the data set
    print(car_insurance_model._data_set.head())  # print again to check the data is shuffled

    car_insurance_model.drop_attribute('Date')  # drop classification attributes for now
    car_insurance_model.drop_attribute('Marital_Status')

    car_insurance_model.drop_all_na()

    car_insurance_model.normalise_data('Sale')

    car_insurance_model.missing_data_ratio_print()  # prints the number of missing values in each column
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)

    #  car_insurance_model._data_set.to_csv('Data_Out/missing_values_dropped.csv', index=False)

    car_insurance_model.knn_model('Sale')


if __name__ == "__main__":
    main()
