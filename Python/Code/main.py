import pandas as pd
from Python.Code.Class_Data_Loader import DataLoader


def main():
    model_df = DataLoader(pd.read_csv("Data_In/VehicleData_csv.csv"))  # loads in the data

    print(model_df._dataset.head())
    print(model_df.dim_data())

    # creates two new columns in the data frame for the year and month from the column Financial Year
    model_df.split_month_year("FinancialYear")

    print(model_df._dataset.head())


if __name__ == "__main__":
    main()

