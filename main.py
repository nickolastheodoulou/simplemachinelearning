import pandas as pd
from Test_Modularised_Code import print_file


def main():
    # Load the Pandas libraries with alias 'pd'

    train = pd.read_csv("Data/train.csv")  # read in train data from the csv file
    test = pd.read_csv("Data/test.csv")

    print_file(train)

if __name__ == "__main__":
    main()