import pandas as pd
from Python.Code.Class_Data_Exploration import DataExploration


def main():
    model_df = DataExploration(pd.read_csv("Data_In/VehicleData_csv.csv"))  # loads in the data

    print(model_df._dataset.head())  # prints the first 5 columns of the data-set
    print(model_df.dim_data())  # prints the dimension of the data set
    model_df.missing_data_ratio_print() #  prints the percentage of missing values in the data set (NONE FOUND!)


    # method that creates two new columns in the data frame for the year and month from the column Financial Year
    model_df.split_month_year("FinancialYear")

    print(model_df._dataset.head())  # prints the first 5 columns of the data-set after adding 2 new columns

    model_df._dataset.to_csv("Data_Out/out.csv") #  saves the pandas data frame to a CSV file

    model_df.box_plot("ConditionScore", "Year")  # box plot of the Condition score for each Year
    model_df.box_plot("ConditionScore", "Month")  # box plot of the Condition score for each Month


if __name__ == "__main__":
    main()

