import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from Class_Data_Loader import DataLoader


class DataExplorer(DataLoader):
    def __init__(self, data_set):
        super().__init__(data_set)

    # method that prints the number of different values in a given column
    def attribute_value_count(self, attribute):
        #  first counts the number of different values in each column then sorts it in ascending order
        my_attribute_count = self._data_set[attribute].value_counts().sort_index()
        print("The count of the different variables in the attribute: ", attribute, ' is\n', my_attribute_count)

    # method that creates a bar graph to show the distribution of an attribute
    def bar_graph_attribute(self, attribute):
        #  first counts the number of different values in each column then sorts it in ascending order
        column_count = self._data_set[attribute].value_counts().sort_index()
        # set the y axis to the values within the series object as a percentage
        y = column_count.values * 100 / column_count.sum()
        x = column_count.index  # set the x axis to the index of the series object
        width_of_bar = 1 / 1.5
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        plt.bar(x, y, width_of_bar, color="#2b8cbe", edgecolor='black')  # plots the bar graph
        #  plt.title('Bar graph of ' + str(column_count.name) + ' Against ' + str(' Sample Size'), fontsize=20)
        plt.xlabel(column_count.name, fontsize=15)  # sets the xlabel to the name of the series object
        plt.ylabel('Percent', fontsize=15)
        plt.xticks(x, rotation=90)  # rotates ticks by 90deg so larger font can be used
        plt.tick_params(labelsize=12)  # increases font of the ticks
        #  file name defined by attribute user input and type of graph
        plt.savefig('Data_Out/' + attribute + '_bar_graph_percentage.pdf', index=False, bbox_inches='tight')
        plt.show()

    # method that prints the number of different values in each column then sorts it in ascending order per
    # classification
    def attribute_value_count_by_classification(self, attribute):
        my_attribute_classification_count = pd.crosstab(self._data_set[attribute], self._data_set.Sale)
        print("The classification count of the different variables in the attribute: ", attribute, ' is\n',
              my_attribute_classification_count)

    def bar_graph_attribute_by_classification(self, attribute):
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        #  Computes a cross-tabulation of user inputted attribute to plot sale and no sale count
        my_attribute_sale_no_sale_matrix = pd.crosstab(self._data_set[attribute], self._data_set['Sale'])

        #  counts number of sales for inputted attribute
        my_attribute_sales_count = my_attribute_sale_no_sale_matrix[self._data_set['Sale'].unique()[0]].values
        #  counts number of no sales for inputted attribute
        my_attribute_no_sales_count = my_attribute_sale_no_sale_matrix[self._data_set['Sale'].unique()[1]].values
        x = my_attribute_sale_no_sale_matrix.index  # set the x axis to index of user inputted attribute

        width = 0.5  # the width of the bars
        #  creates the 2 bars, bottom indicates which bar goes below the current bar
        attribute_sale_bars = plt.bar(x, my_attribute_sales_count, width, edgecolor='black')
        attribute_no_sale_bars = plt.bar(x, my_attribute_no_sales_count, width, bottom=my_attribute_sales_count,
                                         edgecolor='black', )

        # plt.grid()
        plt.ylabel('Count', fontsize=12)
        plt.xlabel(attribute, fontsize=12)
        plt.xticks(x, rotation=90, fontsize=10)  # rotates ticks by 30deg so larger font can be used
        #  create the legend
        plt.legend((attribute_sale_bars[0], attribute_no_sale_bars[0]), ('Sale', ' No Sale'))
        plt.savefig('Data_Out/' + attribute + '_triple_stacked_bar_graph.pdf', index=False, bbox_inches='tight')
        plt.show()

    # method that prints a summary of the distribution of the data
    def describe_attribute(self, attribute):
        description_of_attribute = self._data_set[attribute].describe()
        print("A summary of the disribution for the attribute", attribute, ' is\n', description_of_attribute)

    #  method that prints the percentage of missing data of each column
    def missing_data_ratio_print(self):
        #  define the percentage as the number of missing values in each column/ number of entries * 100
        percent_of_missing_data_in_each_column = ((self._data_set.isnull().sum() / (len(self._data_set))) * 100)

        #  sorts percent_of_missing_data_in_each_column into descending order to be printed
        percent_of_missing_data_in_each_column = percent_of_missing_data_in_each_column.drop(
            percent_of_missing_data_in_each_column[percent_of_missing_data_in_each_column == 0].index).sort_values(
            ascending=False)[:self._data_set.shape[1]]

        #  redefines percent_of_missing_data_in_each_column as a DataFrame with the column head 'Missing Ratio'
        percent_of_missing_data_in_each_column = pd.DataFrame({'Missing Ratio': percent_of_missing_data_in_each_column})
        #  rename the column heading
        percent_of_missing_data_in_each_column = percent_of_missing_data_in_each_column.rename(columns={
            percent_of_missing_data_in_each_column.columns[0]: "Percentage of missing values"})

        #  print the data fame
        print(percent_of_missing_data_in_each_column.head(20))

    #  method that p
    def missing_data_ratio_bar_graph(self):
        #  define the percentage as the number of missing values in each column/ number of entries * 100
        percent_of_missing_data_in_each_column = ((self._data_set.isnull().sum() / (len(self._data_set))) * 100)

        #  sorts percent_of_missing_data_in_each_column into descending order to be printed
        percent_of_missing_data_in_each_column = percent_of_missing_data_in_each_column.drop(
            percent_of_missing_data_in_each_column[percent_of_missing_data_in_each_column == 0].index).sort_values(
                ascending=False)[:self._data_set.shape[1]]

        plt.xticks(rotation='90', fontsize=14)
        # use seaborn package to plot the bar graph
        sns.barplot(x=percent_of_missing_data_in_each_column.index, y=percent_of_missing_data_in_each_column)
        plt.xlabel('Attribute', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.savefig('Data_Out/percentage_of_missing_data.pdf', index=False, bbox_inches='tight')  # save the plot
        plt.show()

    def scatter_plot(self, my_y_attribute, my_x_attribute):
        x = self._data_set[my_x_attribute].values
        # defines the sold price so that it can be loaded into the function each time rather than loading the whole
        # train matrix
        y = self._data_set[my_y_attribute].values
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        plt.scatter(x, y, c="g", alpha=0.5, marker="s")  # scatter plot of the sold price and user chosen attribute
        plt.title('Scatter graph of ' + str(my_y_attribute) + ' against ' + str(my_x_attribute))
        plt.xlabel(my_x_attribute)
        plt.ylabel(my_y_attribute)
        plt.show()

    def scatter_plot_by_classification(self, my_y_attribute, my_x_attribute):
        #  create empty list to store colour with sale being blue and no sale being red
        list_colour_corresponding_to_sale = ['Null'] * len(self._data_set)

        for i in range(0, len(self._data_set)):

            # if a value in the attribute sale is 0 then set the colour to red
            if self._data_set['Sale'].values[i] == 0:
                list_colour_corresponding_to_sale[i] = "r"

            # if a value in the attribute sale is 1 then set the colour to blue
            elif self._data_set['Sale'].values[i] == 1:
                list_colour_corresponding_to_sale[i] = "b"
            else:
                print("Error")
                break

        plt.scatter(self._data_set[my_x_attribute].values, self._data_set[my_y_attribute].values,
                    color=list_colour_corresponding_to_sale)

        plt.xlabel(my_x_attribute)
        plt.ylabel(my_y_attribute)

        # plot empty lists with the desired size and label to create the legend (not possible any other way)
        plt.scatter([], [], c=['r'], alpha=1, label='Sale')
        plt.scatter([], [], c=['b'], alpha=1, label='NoSale')

        plt.legend(loc=0, scatterpoints=1, frameon=False, labelspacing=0, title='Sale or no Sale: ', prop={'size': 9})
        plt.show()

    def histogram_and_q_q(self, attribute):
        x_sigma = self._data_set[attribute].values.std()  # standard deviation
        x_max = self._data_set[attribute].values.max()  # max value
        x_min = self._data_set[attribute].values.min()  # min value
        n = self._data_set[attribute].shape[0]  # number of data points

        # formula to give the number of bins for any dataset
        number_bins = (x_max - x_min) * n ** (1 / 3) / (3.49 * x_sigma)
        number_bins = int(number_bins)  # floors the double to int
        # values being plotted into the histogram
        attribute_being_plotted = self._data_set[attribute].values

        # defined y to find y max to place the text
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        y, i, _ = plt.hist(attribute_being_plotted, density=True, bins=number_bins, facecolor='paleturquoise',
                           alpha=0.75, edgecolor='black', linewidth=1.2,
                           label='Histogram: (Skewness: ' + "{0:.3f}".format(self._data_set[attribute].skew()) +
                                 ' and Kurtosis: ' + "{0:.3f}".format(self._data_set[attribute].kurt()) + ')')

        x = np.linspace(x_min, x_max, len(attribute_being_plotted))

        # Get the fitted parameters used by the function for the normal distribution
        (mu, sigma) = stats.norm.fit(self._data_set[attribute])
        normal_distribution = stats.norm.pdf(x, mu, sigma)  # define the norml distribution in terms of x, mu and sigma
        plt.plot(x, normal_distribution, 'k', linewidth=2,
                 label='Normal distribution: ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))

        plt.legend(loc='best')  # adds the legend
        plt.ylabel('Probability')
        plt.xlabel(attribute)
        plt.title('Histogram of ' + str(attribute))
        plt.show()

        stats.probplot(self._data_set[attribute], plot=plt)  # Q-Q plot
        plt.title('Quantile-Quantile plot of ' + attribute)
        plt.show()
