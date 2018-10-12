from Python.Code.Class_Data_Loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataExploration(DataLoader):  # inherits the members test and train from data_matrix
    def __init__(self, data_set):
        super().__init__(data_set)

    #  function that prints the percentage of missing data of each column
    def missing_data_ratio_print(self):
        data_set_missing = ((self._dataset.isnull().sum() / (len(self._dataset))) * 100)
        data_set_missing = data_set_missing.drop(data_set_missing[data_set_missing == 0].index).sort_values(
            ascending=False)[:self._dataset.shape[1]]
        missing_data = pd.DataFrame({'Missing Ratio': data_set_missing})
        missing_data = missing_data.rename(columns={missing_data.columns[0]: "your value"})
        print(missing_data.head(20))



    # function that creates a box plot
    def box_plot(self, target, attribute):
        data = pd.concat([self._dataset[target], self._dataset[attribute]], axis=1)  # defines the data
        plt.subplots(figsize=(16, 8))  # changes the size of the fig
        fig = sns.boxplot(x=attribute, y=target, data=data)
        fig.axis(ymin=0, ymax=self._dataset[target].values.max())  # defines the y axis
        plt.xticks(rotation=90)  # rotates the x ticks so that they are easier to read when the strings are longer
        #  plt.savefig('Plots/boxplot.png', index=False)
        plt.show()