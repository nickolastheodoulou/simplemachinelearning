import pandas as pd
from Class_Data_Exploration import DataExploration


def main():
    model_df = DataExploration(pd.read_csv("Data_In/VehicleData_csv.csv"))  # loads in the data

    print(model_df._data_set.head())  # prints the first 5 columns of the data-set
    print(model_df.dim_data())  # prints the dimension of the data set
    model_df.missing_data_ratio_print()  # prints the percentage of missing values in the data set (NONE FOUND!)

    #  calls a function to combine the column VehicleType and Manufacturer to a new column named
    #  ManufacturerAndVehicleType
    model_df.combine_columns("ManufacturerAndVehicleType", "VehicleType", "Manufacturer")

    print(model_df._data_set.head())  # print the data_set to check it worked correctly

    #  model_df._data_set.to_csv("Data_Out/data_set_out.csv")  # saves the pandas data frame to a CSV file

    model_df.box_plot("ConditionScore", "FinancialYear")  # box plot of the Condition score for each Year
    model_df.box_plot("ConditionScore", "Manufacturer")  # box plot of the Condition score for each Manufacturer

    model_df.box_plot("ConditionScore", "ManufacturerAndVehicleType")  # box plot of the Condition score for each Year

    # define a variable as the return of returns the number of different values in the financial year.
    financial_year_count = model_df.column_value_count("FinancialYear")

    #  prints the number of inspections each year
    print(financial_year_count)
    #  prints some of the statistics for the number of inspections each year
    print(financial_year_count.describe())

    #  drops some of the years to inspect the statistics when certain values are dropped
    financial_year_count = financial_year_count.drop(financial_year_count.index[[0, 9, 10, 11]])
    print(financial_year_count)
    print(financial_year_count.describe())

    #  function that plots and saves the percentage of what year each inspection occurs.
    model_df.bar_graph_percentage("FinancialYear")

    #  function that plots the percentage difference of what year each inspection occurs.
    model_df.line_graph_percentage_difference("FinancialYear")

    model_df.triple_stacked_bar_graph('FinancialYear', 'VehicleType')


if __name__ == "__main__":
    main()
