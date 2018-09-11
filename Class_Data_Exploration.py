import matplotlib.pyplot as plt
import numpy as np
from Class_Data_Loader import DataLoader
from scipy import stats
import seaborn as sns
import pandas as pd


class DataExploration(DataLoader):  # inherits the members test and train from data_matrix
    def __init__(self, train_X, test_X):
        super().__init__(train_X, test_X)

    def sale_price_against_attribute_scatter_plot(self, target, attribute):  # method that plots sales against an attribute
        x = self._train_X[attribute].values
        y = self._train_X[target].values  # defines the sold price so that it can be loaded into the function each time rather than loading the whole train matrix
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        plt.scatter(x, y, c="g", alpha=0.5, marker="s")  # scatter plot of the sold price and user chosen attribute
        plt.title('Scatter graph of ' + str(target) + ' against ' + str(attribute))
        plt.xlabel(attribute)
        plt.ylabel(target)
        plt.show()

    def describe_attribute(self, attribute):
        print(self._train_X[attribute].describe())

    def histogram_and_q_q(self, attribute):
        x_sigma = self._train_X[attribute].values.std()  # standard deviation
        x_max = self._train_X[attribute].values.max()  # max value
        x_min = self._train_X[attribute].values.min()  # min value
        n = self._train_X[attribute].shape[0]  # number of datapoints

        number_bins = (x_max - x_min) * n ** (1 / 3) / (3.49 * x_sigma)  # formula to give the number of bins for any dataset
        number_bins = int(number_bins)  # floors the double to int
        attribute_being_plotted = self._train_X[attribute].values#values being plotted into the histogram

        # defined y to find y max to place the text
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        y, i, _ = plt.hist(attribute_being_plotted, density=True, bins=number_bins, facecolor='paleturquoise',
            alpha=0.75,edgecolor='black', linewidth=1.2,
                label='Histogram: (Skewness: ' + "{0:.3f}".format(self._train_X[attribute].skew()) +
                        ' and Kurtosis: ' + "{0:.3f}".format(self._train_X[attribute].kurt())+')')


        x = np.linspace(x_min, x_max, len(attribute_being_plotted))
        param = stats.rayleigh.fit(attribute_being_plotted)  # distribution fitting
        pdf_fitted = stats.rayleigh.pdf(x, loc=param[0], scale=param[1])  # fitted distribution
        plt.plot(x, pdf_fitted, 'cornflowerblue', label='Rayleigh distribution')#plots the fit


        (mu, sigma) = stats.norm.fit(self._train_X[attribute]) # Get the fitted parameters used by the function for the normal distribution
        normal_distribution = stats.norm.pdf(x, mu, sigma) #  define the norml distribution in terms of x, mu and sigma
        plt.plot(x, normal_distribution, 'k', linewidth=2 ,label='Normal distribution: ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))

        plt.legend(loc='best')#adds the legend
        plt.ylabel('Probability')
        plt.xlabel(attribute)
        plt.title('Histogram of ' + str(attribute))
        plt.show()

        stats.probplot(self._train_X[attribute], plot=plt)  # Q-Q plot
        plt.title('Quantile-Quantile plot of ' + attribute)
        plt.show()

    def boxplot(self, attribute, target):# box plot overallqual/salepricedata = pd.concat([matrix._train_X[target], matrix._train_X[attribute]], axis=1)  # defines the data
        data = pd.concat([self._train_X[target], self._train_X[attribute]], axis=1)  # defines the data
        plt.subplots(figsize=(16, 8))#changes the size of the fig
        fig = sns.boxplot(x=attribute, y=target, data=data)
        fig.axis(ymin=0, ymax=self._train_X[target].values.max())  # defines the y axis
        plt.xticks(rotation=90)  # rotates the x ticks so that they are easier to read when the strings are longer
        #  plt.savefig('Plots/boxplot.png', index=False)
        plt.show()

    def heatmap(self):
        correlation_matrix = self._train_X.corr()  # correlation matrix
        plt.subplots(figsize=(12, 9))#size of fig
        z_text = np.around(correlation_matrix, decimals=1)  # Only show rounded value (full value on hover)
        sns.heatmap(z_text, vmax=.8, square=True, annot=True, fmt='.1f', annot_kws={'size': 7})  # creates the heatmap
        plt.savefig('Plots/heatmap.svg', format='svg', index=False)
        plt.show()
        return correlation_matrix

    def heatmap_correlated_attributes(self, number_of_highest_correlated_attributes, target):#most correlated attributes to the target
        correlation_matrix = self._train_X.corr()  # correlation matrix
        cols = correlation_matrix.nlargest(number_of_highest_correlated_attributes, target)[target].index
        correlation_matrix_highest_correlated = np.corrcoef(self._train_X[cols].values.T)  # saleprice correlation matrix
        sns.set(font_scale=1.25)
        sns.heatmap(correlation_matrix_highest_correlated, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)
        #annot includes the number within the graph, fmt set to two decimal places, annot_kws is the size of font inside the plot
        plt.show()

    def missing_data_ratio_and_bar_graph(self, target):
        train_X_na = self._train_X
        train_X_na.drop([target], axis=1, inplace=True)
        train_X_na = ((self._train_X.isnull().sum() / (len(self._train_X))) * 100)
        train_X_na = train_X_na.drop(train_X_na[train_X_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio': train_X_na})
        missing_data = missing_data.rename(columns={missing_data.columns[0]: "your value"})
        print(missing_data.head(20))

        plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=train_X_na.index, y=train_X_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.show()
