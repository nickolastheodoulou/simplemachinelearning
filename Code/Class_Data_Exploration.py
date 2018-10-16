import matplotlib.pyplot as plt
import numpy as np
from Code.Class_Data_Loader import DataLoader
from scipy import stats
import seaborn as sns
import pandas as pd


# inherits the members test and train from data_matrix
class DataExploration(DataLoader):
    def __init__(self, train_x, test_x):
        super().__init__(train_x, test_x)

    # function that returns the number of different values in a given column
    def column_value_count(self, attribute):
        #  first counts the number of different values in each column then sorts it in ascending order
        column_count = self._train_x[attribute].value_counts().sort_index()
        return column_count

    def describe_attribute(self, attribute):
        print(self._train_x[attribute].describe())

    # function that plots sales against an attribute
    def scatter_plot(self, target, attribute):
        x = self._train_x[attribute].values
        # defines the sold price so that it can be loaded into the function each time rather than loading the whole
        # train matrix
        y = self._train_x[target].values
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        plt.scatter(x, y, c="g", alpha=0.5, marker="s")  # scatter plot of the sold price and user chosen attribute
        plt.title('Scatter graph of ' + str(target) + ' against ' + str(attribute))
        plt.xlabel(attribute)
        plt.ylabel(target)
        plt.show()

    def histogram_and_q_q(self, attribute):
        x_sigma = self._train_x[attribute].values.std()  # standard deviation
        x_max = self._train_x[attribute].values.max()  # max value
        x_min = self._train_x[attribute].values.min()  # min value
        n = self._train_x[attribute].shape[0]  # number of data points

        # formula to give the number of bins for any dataset
        number_bins = (x_max - x_min) * n ** (1 / 3) / (3.49 * x_sigma)
        number_bins = int(number_bins)  # floors the double to int
        # values being plotted into the histogram
        attribute_being_plotted = self._train_x[attribute].values

        # defined y to find y max to place the text
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        y, i, _ = plt.hist(attribute_being_plotted, density=True, bins=number_bins, facecolor='paleturquoise',
                           alpha=0.75, edgecolor='black', linewidth=1.2,
                           label='Histogram: (Skewness: ' + "{0:.3f}".format(self._train_x[attribute].skew()) +
                                 ' and Kurtosis: ' + "{0:.3f}".format(self._train_x[attribute].kurt()) + ')')

        x = np.linspace(x_min, x_max, len(attribute_being_plotted))
        param = stats.rayleigh.fit(attribute_being_plotted)  # distribution fitting
        pdf_fitted = stats.rayleigh.pdf(x, loc=param[0], scale=param[1])  # fitted distribution
        plt.plot(x, pdf_fitted, 'cornflowerblue', label='Rayleigh distribution')  # plots the fit

        # Get the fitted parameters used by the function for the normal distribution
        (mu, sigma) = stats.norm.fit(self._train_x[attribute])
        normal_distribution = stats.norm.pdf(x, mu, sigma)  # define the norml distribution in terms of x, mu and sigma
        plt.plot(x, normal_distribution, 'k', linewidth=2,
                 label='Normal distribution: ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))

        plt.legend(loc='best')  # adds the legend
        plt.ylabel('Probability')
        plt.xlabel(attribute)
        plt.title('Histogram of ' + str(attribute))
        plt.show()

        stats.probplot(self._train_x[attribute], plot=plt)  # Q-Q plot
        plt.title('Quantile-Quantile plot of ' + attribute)
        plt.show()


    def box_plot(self, target, attribute):
        #  sort the dataset into acending order of the attribute to be plotted
        self._train_x = self._train_x.sort_values([attribute]).reset_index(drop=True)
        data_in = pd.concat([self._train_x[target], self._train_x[attribute]], axis=1)  # defines the data
        plt.subplots(figsize=(16, 8))  # changes the size of the fig

        fig = sns.boxplot(x=attribute, y=target, data=data_in)
        fig.set_xlabel(attribute, fontsize=12)
        fig.set_ylabel(target, fontsize=12)
        fig.axis(ymin=0, ymax=self._train_x[target].values.max())  # defines the y axis
        plt.xticks(rotation=90)  # rotates the x ticks so that they are easier to read when the strings are longer
        plt.tick_params(labelsize=12)

        #  file name defined by attribute user input and type of graph
        plt.savefig('Data_Out/' + attribute + '_boxplot.pdf', index=False, bbox_inches='tight')
        plt.show()

    def heatmap(self):
        correlation_matrix = self._train_x.corr()  # correlation matrix
        plt.subplots(figsize=(12, 9))  # size of fig
        z_text = np.around(correlation_matrix, decimals=1)  # Only show rounded value (full value on hover)
        sns.heatmap(z_text, vmax=.8, square=True, annot=True, fmt='.1f', annot_kws={'size': 7})  # creates the heatmap
        plt.savefig('Plots/heatmap.svg', format='svg', index=False)
        plt.show()
        return correlation_matrix

    # function that plots most correlated attributes to the target
    def heatmap_correlated_attributes(self, number_of_highest_correlated_attributes, target):
        correlation_matrix = self._train_x.corr()  # correlation matrix
        cols = correlation_matrix.nlargest(number_of_highest_correlated_attributes, target)[target].index
        # saleprice correlation matrix
        correlation_matrix_highest_correlated = np.corrcoef(self._train_x[cols].values.T)
        sns.set(font_scale=1.25)
        sns.heatmap(correlation_matrix_highest_correlated, cbar=True, annot=True, square=True, fmt='.2f',
                    annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)
        #  annot includes the number within the graph, fmt set to two decimal places, annot_kws is the size of font
        # inside the plot
        plt.show()

    ####################################################################################################################
    #  need to combine these 3 and come up with a better way to structure train_x and test_x

    #  function that prints the percentage of missing data of each column
    def missing_data_ratio_print(self):
        data_set_missing = ((self._train_x.isnull().sum() / (len(self._train_x))) * 100)
        data_set_missing = data_set_missing.drop(data_set_missing[data_set_missing == 0].index).sort_values(
            ascending=False)[:self._train_x.shape[1]]
        missing_data = pd.DataFrame({'Missing Ratio': data_set_missing})
        missing_data = missing_data.rename(columns={missing_data.columns[0]: "your value"})
        print(missing_data.head(20))

    def train_missing_data_ratio_and_bar_graph(self):
        train_x_missing = ((self._train_x.isnull().sum() / (len(self._train_x))) * 100)
        train_x_missing = train_x_missing.drop(train_x_missing[train_x_missing == 0].index).sort_values(ascending=False)[:self._train_x.shape[1]]
        missing_data = pd.DataFrame({'Missing Ratio': train_x_missing})
        missing_data = missing_data.rename(columns={missing_data.columns[0]: "your value"})
        print(missing_data.head(20))

        plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=train_x_missing.index, y=train_x_missing)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.show()

    def test_missing_data_ratio_and_bar_graph(self):
        test_x_missing = ((self._test_x.isnull().sum() / (len(self._test_x))) * 100)
        test_x_missing = test_x_missing.drop(test_x_missing[test_x_missing == 0].index).sort_values(ascending=False)[:self._test_x.shape[1]]
        missing_data = pd.DataFrame({'Missing Ratio': test_x_missing})
        missing_data = missing_data.rename(columns={missing_data.columns[0]: "your value"})
        print(missing_data.head(20))

        plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=test_x_missing.index, y=test_x_missing)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.show()

    ####################################################################################################################

    # function that creates a bar graph of each different value as a percentage for a given attribute
    def bar_graph_percentage(self, attribute):
        #  first counts the number of different values in each column then sorts it in ascending order
        column_count = self._train_x[attribute].value_counts().sort_index()

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

    def line_graph_percentage_difference(self, attribute):
        #  counts the number of attributes and store
        attribute_list_count = self._train_x[attribute].value_counts().sort_index()

        #  define the list of attributes in ascending order
        attribute_list = attribute_list_count.index

        # define empty numpy array to store the y coordinates which will be the percentage difference
        y = np.zeros(attribute_list_count.count() - 1)
        #  define empty list to store the ticks which will be combined from attribute_list of size attribute_list - 1
        my_x_ticks = [0] * (attribute_list_count.count() - 1)

        # for loop to calculate the percentage difference and the value of the x ticks
        for i in range(0, attribute_list_count.count() - 1):
            y[i] = ((attribute_list_count.values[i + 1] - attribute_list_count.values[i]) /
                    attribute_list_count.values[i]) * 100

            my_x_ticks[i] = str(attribute_list.values[i]) + " to " + str(attribute_list.values[i + 1])

        x = attribute_list_count[:-1].index  # drops the final column in the series object and sets x to the index
        plt.grid()
        plt.plot(x, y, c="g", alpha=0.5, marker="s")
        #  plt.title('Bar graph of ' + str(column_count.name) + ' Against ' + str(' Sample Size'), fontsize=20)
        plt.xlabel(attribute, fontsize=12)  # sets the xlabel to the name of the series object
        plt.ylabel('Percentage Change', fontsize=12)
        plt.xticks(x, my_x_ticks, rotation=90)  # rotates ticks by 90deg so larger font can be used
        plt.tick_params(labelsize=12)  # increases font of the ticks

        #  file name defined by attribute user input and type of graph
        plt.savefig('Data_Out/'+attribute + '_line_graph_percentage_difference.pdf', index=False, bbox_inches='tight')
        plt.show()

    #  function that plots a triple stacked bar graph of attribute as the x coordinate and sub-attribute as the
    # attribute to be stacked within the bar (Needs to be generalised!
    def triple_stacked_bar_graph(self, attribute, sub_attribute):
        #  Computes a cross-tabulation of two user inputted attributes
        attribute_and_sub_attribute_matrix = pd.crosstab(self._train_x[attribute], self._train_x[sub_attribute])

        #  defines the y values of the three sub attributes as the three columns in column_count matrix
        sub_attribute_one_count = attribute_and_sub_attribute_matrix[self._train_x[sub_attribute].unique()[0]].values
        sub_attribute_two_count = attribute_and_sub_attribute_matrix[self._train_x[sub_attribute].unique()[1]].values
        sub_attribute_three_count = attribute_and_sub_attribute_matrix[self._train_x[sub_attribute].unique()[2]].values

        x = attribute_and_sub_attribute_matrix.index  # set the x axis to the year which is the intex of
        #  attribute_and_sub_attribute_matrix

        width = 0.35  # the width of the bars

        #  creates the 3 bars, bottom indicates which bar goes below the current bar
        sub_attribute_one_bar = plt.bar(x, sub_attribute_one_count, width)
        sub_attribute_two_bar = plt.bar(x, sub_attribute_two_count, width, bottom=sub_attribute_one_count, )
        sub_attribute_three_bar = plt.bar(x, sub_attribute_three_count, width, bottom=sub_attribute_two_count
                                                                                      + sub_attribute_one_count, )

        plt.grid()
        plt.ylabel('Number Of Samples', fontsize=14)
        plt.xlabel(attribute, fontsize=14)
        plt.xticks(x, rotation=30)  # rotates ticks by 30deg so larger font can be used
        #  create the legend of each bar by the corresponding sub_attribute which is the ith unique value in the
        #  column VehicleType within self._data_set
        plt.legend((sub_attribute_one_bar[0], sub_attribute_two_bar[0], sub_attribute_three_bar[0]),
                   (self._train_x.VehicleType.unique()[0], self._train_x.VehicleType.unique()[1],
                    self._train_x.VehicleType.unique()[2]))

        plt.savefig('Data_Out/' + attribute + '_triple_stacked_bar_graph.pdf', index=False, bbox_inches='tight')
        plt.show()