import pandas as pd
import matplotlib.pyplot as plt
from Class_Data_Loader import data_matrix


class Data_Exploration(data_matrix):  # inherits the members test and train from data_matrix
    def sale_price_against_attribute_scatter_plot(self, attribute):#method that plots sales against an attribute
        x = self._train_X[attribute].values
        Sold_Price = self._train_Y  # defines the sold price so that it can be loaded into the function each time rather than loading the whole train matrix
        plt.scatter(x, Sold_Price, c="g", alpha=0.25,label="")  # scatter plot of the sold price and user chosen attribute
        plt.xlabel(attribute)
        plt.ylabel("Sold Price of House")
        # plt.legend(loc=2)
        plt.show()

