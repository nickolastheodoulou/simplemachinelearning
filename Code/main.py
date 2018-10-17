import pandas as pd
from Code.Class_Data_Explorer import DataExploration


def main():
    car_insurance_model = DataExploration(pd.read_csv("Data_In/DS_Assessment.csv"))  # first load in the data

    # print the dimension of the data set
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)

    car_insurance_model.column_value_count('Age')  # counts the number of different values in the 'Age' column
    # displays and saves a bargraph showing the percentage of each value for the column Age in the data set
    car_insurance_model.bar_graph_distribution('Age')

    car_insurance_model.describe_attribute('Age')  # prints a summary of the distribution of the column 'Age'
    car_insurance_model.missing_data_ratio_print()  # prints the number of missing values in each column

    # displays and saves a bargraph of the percentage of missing values
    car_insurance_model.missing_data_ratio_bar_graph()

    print(car_insurance_model._data_set.head())  # prints the first 5 columns of the data set
    car_insurance_model.shuffle_data_set()  # shuffle the data set
    print(car_insurance_model._data_set.head())  # print again to check the data is shuffled


if __name__ == "__main__":
    main()
