import pandas as pd

from Class_Data_Explorer import DataExplorer


def main():
    # Main used for DfBL data set

    model_DfBL = DataExplorer(pd.read_csv("Data_In/DfBL/VehicleData_csv.csv"))  # loads in the data

    print(model_DfBL._data_set.head())  # prints the first 5 columns of the data-set
    print(model_DfBL._data_set.shape)  # prints the dimension of the data set
    model_DfBL.missing_data_ratio_print()  # prints the percentage of missing values in the data set (NONE FOUND!)

    #  calls a function to combine the column VehicleType and Manufacturer to a new column named
    #  ManufacturerAndVehicleType
    model_DfBL.combine_columns("ManufacturerAndVehicleType", "VehicleType", "Manufacturer")

    print(model_DfBL._data_set.head())  # print the data_set to check it worked correctly

    #  model_DfBL._data_set.to_csv("Data_Out/data_set_out.csv")  # saves the pandas data frame to a CSV file

    model_DfBL.box_plot("ConditionScore", "FinancialYear")  # box plot of the Condition score for each Year
    model_DfBL.box_plot("ConditionScore", "Manufacturer")  # box plot of the Condition score for each Manufacturer

    model_DfBL.box_plot("ConditionScore", "ManufacturerAndVehicleType")  # box plot of the Condition score for each Year

    # function that prints the number of inspections each financial year.
    model_DfBL.attribute_value_count("FinancialYear")

    #  function that plots ans saves the percentage of what year each inspection occurs.
    model_DfBL.bar_graph_attribute("FinancialYear")

    #  function that plots the percentage difference of what year each inspection occurs.
    model_DfBL.line_graph_percentage_difference("FinancialYear")

    model_DfBL.triple_stacked_bar_graph('FinancialYear', 'VehicleType')

    ####################################################################################################################


if __name__ == "__main__":
    main()