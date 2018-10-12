import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Python.Code.Class_Data_Exploration import DataExploration


def main():
    model_df = DataExploration(pd.read_csv("Data_In/VehicleData_csv.csv"))  # loads in the data

    print(model_df._data_set.head())  # prints the first 5 columns of the data-set
    print(model_df.dim_data())  # prints the dimension of the data set
    model_df.missing_data_ratio_print()  # prints the percentage of missing values in the data set (NONE FOUND!)

    model_df.year_new_column("FinancialYear")  # create a new column for the year named: "Year"
    model_df._data_set.to_csv("Data_Out/data_set_out.csv")  # saves the pandas data frame to a CSV file

    model_df.box_plot("ConditionScore", "Year")  # box plot of the Condition score for each Year
    model_df.box_plot("ConditionScore", "Manufacturer")  # box plot of the Condition score for each Manufacturer

    # function that prints the number of inspections each financial year.
    model_df.column_value_count("FinancialYear")

    #  function that plots the percentage of what year each inspection occurs.
    model_df.bar_graph_percentage("Year")

###################################################################################
# need to move into function within exploration
    FinancialYearCount = model_df._data_set["FinancialYear"].value_counts().sort_index()

    percentage_change = np.zeros(FinancialYearCount.count() - 1)

    for i in range(0, FinancialYearCount.count() - 1):  # for loop to calculate the percentage difference
        percentage_change[i] = ((FinancialYearCount.values[i+1] - FinancialYearCount.values[i]) / FinancialYearCount.values[i]) * 100

    print(percentage_change)


    x = FinancialYearCount[:-1].index #drops the final column in the series object and sets x to the index
    plt.subplots(figsize=(16, 8))  # changes the size of the fig
    plt.grid()
    plt.plot(x, percentage_change, c="g", alpha=0.5, marker="s")
    #  plt.title('Bar graph of ' + str(column_count.name) + ' Against ' + str(' Sample Size'), fontsize=20)
    plt.xlabel(FinancialYearCount.name, fontsize=24)  # sets the xlabel to the name of the series object
    plt.ylabel('Percent Change', fontsize=24)
    plt.xticks(x, rotation=30)  # rotates ticks by 90deg so larger font can be used
    plt.tick_params(labelsize=22)  # increases font of the ticks
    plt.show()

##########################################################################################

if __name__ == "__main__":
    main()
