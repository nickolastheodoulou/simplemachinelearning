import matplotlib.pyplot as plt
from Class_Data_Loader import DataLoader


class DataExploration(DataLoader):  # inherits the members test and train from data_matrix
    def __init__(self, train_X, test_X,  train_Y, test_Y):
        super().__init__(train_X, test_X, train_Y, test_Y)

    def sale_price_against_attribute_scatter_plot(self, target, attribute):  # method that plots sales against an attribute
        x = self._train_X[attribute].values
        y = self._train_Y  # defines the sold price so that it can be loaded into the function each time rather than loading the whole train matrix
        plt.scatter(x, y, c="g", alpha=0.25, label="")  # scatter plot of the sold price and user chosen attribute
        plt.xlabel(attribute)
        plt.ylabel(target)
        # plt.legend(loc=2)
        plt.show()

    def describe_target(self):
        print(self._train_Y.describe())

    def describe_attribute(self, attribute):
        print(self._train_X[attribute].describe())
