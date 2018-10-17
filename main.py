import pandas as pd


def main():
    print("Hello World")
    data_set = pd.read_csv("Data_In/DS_Assessment.csv")
    print(data_set.head())


if __name__ == "__main__":
    main()
