import pandas as pd
import matplotlib.pyplot as plt

def sale_price_against_attribute_scatter_plot(attribute):
    train = pd.read_csv("Data/train.csv")  # read in train data from the csv file
    x = train[attribute].values
    Sold_Price = train['SalePrice'].values  # defines the sold price so that it can be loaded into the function each time rather than loading the whole train matrix
    plt.scatter(x, Sold_Price, c="g", alpha=0.25, label="")#scatter plot of the sold price and user chosen attribute
    plt.xlabel(attribute)
    plt.ylabel("Sold Price of House")
    # plt.legend(loc=2)
    plt.show()