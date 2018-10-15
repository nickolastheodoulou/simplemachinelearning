from Python.Class_Data_Loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataExploration(DataLoader):  # inherits the members test and train from data_matrix
    def __init__(self, data_set):
        super().__init__(data_set)

    def dim_data(self):  # function that prints the dimensions of the dataset
        print('The dimensions of the dataset is', self._data_set.shape)

    #  function that prints the percentage of missing data of each column
    def missing_data_ratio_print(self):
        data_set_missing = ((self._data_set.isnull().sum() / (len(self._data_set))) * 100)
        data_set_missing = data_set_missing.drop(data_set_missing[data_set_missing == 0].index).sort_values(
            ascending=False)[:self._data_set.shape[1]]
        missing_data = pd.DataFrame({'Missing Ratio': data_set_missing})
        missing_data = missing_data.rename(columns={missing_data.columns[0]: "your value"})
        print(missing_data.head(20))

    # function that prints the number of different values in a given column
    def column_value_count(self, attribute):
        #  first counts the number of different values in each column then sorts it in ascending order
        column_count = self._data_set[attribute].value_counts().sort_index()
        print(column_count)

    # function that creates a bar graph of each different value as a percentage for a given attribute
    def bar_graph_percentage(self, attribute):
        #  first counts the number of different values in each column then sorts it in ascending order
        column_count = self._data_set[attribute].value_counts().sort_index()

        # set the y axis to the values within the series object as a percentage
        y = column_count.values * 100 / column_count.sum()
        x = column_count.index  # set the x axis to the index of the series object
        width_of_bar = 1 / 1.5
        plt.subplots(figsize=(16, 8))  # changes the size of the fig

        plt.bar(x, y, width_of_bar, color="#2b8cbe", edgecolor='black', linewidth=2)  # plots the bar graph
        #  plt.title('Bar graph of ' + str(column_count.name) + ' Against ' + str(' Sample Size'), fontsize=20)
        plt.xlabel(column_count.name, fontsize=24)  # sets the xlabel to the name of the series object
        plt.ylabel('Percent', fontsize=24)
        plt.xticks(x, rotation=30)  # rotates ticks by 90deg so larger font can be used
        plt.tick_params(labelsize=22)  # increases font of the ticks

        #  file name defined by attribute user input and type of graph
        plt.savefig('Data_Out/' + attribute + '_bar_graph_percentage.pdf', index=False, bbox_inches='tight')

        plt.show()

    def bar_graph_percentage_difference(self, attribute):
        #  counts the number of attributes and store
        attribute_list_count = self._data_set[attribute].value_counts().sort_index()

        #  define the list of attributes in ascending order
        attribute_list = attribute_list_count.index

        # define empty numpy array to store the y coordinates which will be the percentage difference
        y = np.zeros(attribute_list_count.count() - 1)
        #  define empty list to store the ticks which will be combined from attribute_list of size attribute_list - 1
        my_x_ticks = [0] * (attribute_list_count.count() - 1)

        for i in range(0, attribute_list_count.count() - 1):  # for loop to calculate the percentage difference
            y[i] = ((attribute_list_count.values[i + 1] - attribute_list_count.values[i]) /
                                    attribute_list_count.values[i]) * 100

            my_x_ticks[i] = str(attribute_list.values[i]) + " to " + str(attribute_list.values[i + 1])

        #  print(str(attribute_list.values[0]) + " to " + str(attribute_list.values[0 + 1]))

        x = attribute_list_count[:-1].index  # drops the final column in the series object and sets x to the index
        plt.grid()
        plt.plot(x, y, c="g", alpha=0.5, marker="s")
        #  plt.title('Bar graph of ' + str(column_count.name) + ' Against ' + str(' Sample Size'), fontsize=20)
        plt.xlabel(attribute, fontsize=12)  # sets the xlabel to the name of the series object
        plt.ylabel('Percentage Change', fontsize=12)
        plt.xticks(x, my_x_ticks, rotation=90)  # rotates ticks by 90deg so larger font can be used
        plt.tick_params(labelsize=12)  # increases font of the ticks

        #  file name defined by attribute user input and type of graph
        plt.savefig('Data_Out/'+attribute + '_bar_graph_percentage_difference.pdf', index=False, bbox_inches='tight')
        plt.show()

    # function that creates a box plot
    def box_plot(self, target, attribute):
        #  sort the dataset into acending order of the attribute to be plotted
        self._data_set = self._data_set.sort_values([attribute]).reset_index(drop=True)
        data_in = pd.concat([self._data_set[target], self._data_set[attribute]], axis=1)  # defines the data
        plt.subplots(figsize=(16, 8))  # changes the size of the fig

        fig = sns.boxplot(x=attribute, y=target, data=data_in)
        fig.set_xlabel(attribute, fontsize=24)
        fig.set_ylabel(target, fontsize=24)
        fig.axis(ymin=0, ymax=self._data_set[target].values.max())  # defines the y axis
        plt.xticks(rotation=90)  # rotates the x ticks so that they are easier to read when the strings are longer
        plt.tick_params(labelsize=24)

        #  file name defined by attribute user input and type of graph
        plt.savefig('Data_Out/' + attribute + '_boxplot.pdf', index=False, bbox_inches='tight')
        plt.show()
