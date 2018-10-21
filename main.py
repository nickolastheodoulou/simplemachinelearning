import pandas as pd
import numpy as np

from Class_Data_Modeler import DataModeler


def main():
    # constructs the object car_insurance_model by loading in the data to the class DataModeler
    car_insurance_model = DataModeler(pd.read_csv("Data_In/DS_Assessment.csv"))

    ####################################################################################################################
    # EXPLORATION
    ####################################################################################################################

    # Get a very broad understanding of the data from the dimension and first 5 rows

    # Prints the dimension of the data after successfully storing the data set as a Pandas data frame
    print("The dimension of the car insurance data is: ", car_insurance_model._data_set.shape)
    print(car_insurance_model._data_set.head())  # prints the first 5 rows of the data set

    ####################################################################################################################
    # Example of some of the graphs used to explore the data for the attribute: Age
    # These methods can be used for any attribute

    # counts number of each age
    car_insurance_model.attribute_value_count('Age')
    # counts Sale or NoSale for each number of age
    car_insurance_model.attribute_value_count_by_classification('Age')
    # displays and saves a bar graph showing the percentage of each value for the attribute Age in the data set
    car_insurance_model.bar_graph_attribute('Age')
    # displays a stacked Sale and NoSale bar graph for each attribute in Age in the data set
    car_insurance_model.bar_graph_attribute_by_classification('Age')
    # prints a summary of the distribution of the column 'Age' such as mean, standard deviation etc
    car_insurance_model.describe_attribute('Age')
    # plots a histogram of the attribute Age and also a quantile quantile plot
    car_insurance_model.histogram_and_q_q('Age')
    # plots a scatter plot of Age and Price
    car_insurance_model.scatter_plot('Age', 'Price')
    # plots a scatter plot of Age and Price for Sale and NoSale
    car_insurance_model.scatter_plot_by_classification('Age', 'Price')


    ####################################################################################################################
    # Observe how much data is missing for each attribute
    car_insurance_model.missing_data_ratio_print()
    # displays and saves a bar graph of the percentage of missing values
    car_insurance_model.missing_data_ratio_bar_graph()

    ####################################################################################################################
    # PROCESSING
    ####################################################################################################################

    # Normalise attributes before imputing
    attributes_to_normalise = ['Veh_Mileage', 'Credit_Score', 'License_Length', 'Veh_Value', 'Price', 'Age', 'Tax']

    for i in attributes_to_normalise:
        car_insurance_model.normalise_attribute(i)

    ####################################################################################################################
    # creating new features from the attribute date

    # decided to add day_of_the_week column to see if any information can be extracted
    car_insurance_model.add_day_of_week_attribute()
    # bar graph of new column to see if any new information can be obtained
    car_insurance_model.bar_graph_attribute_by_classification('days_of_the_week')
    # can see that on Friday typically there are less sales hence decided to create new column

    # one hot encodes the column days_of_the_week by adding 7 new attributes
    car_insurance_model.one_hot_encode_attribute('days_of_the_week')
    # drop date as there are so many different days
    car_insurance_model.drop_attribute('Date')

    ####################################################################################################################
    # Dealing with the attributes Tax and Price

    # scatter plot the two attributes as they appear very highly correlated and could be used to impute the data
    car_insurance_model.scatter_plot_by_classification("Tax", "Price")
    # found that tax and price follow two linear equations using car_insurance_model.scatter_plot("Tax", "Price")
    # the cutoff between following either equation was when the tax was between a value of 32 to 35 which was found by
    # looking through the data set:
    # typically when tax < 34, tax = 0.05 * price and when tax > 34, tax = 0.1 * price
    # hence this can be used to impute missing values more accurately

    # compare how many values are imputed using this method
    car_insurance_model.missing_data_ratio_print()
    car_insurance_model.impute_price()
    car_insurance_model.impute_tax()
    car_insurance_model.missing_data_ratio_print()

    # as only 5 values are missing for both Price and Tax, the mean is imputed for these values
    car_insurance_model.impute_mean('Price')
    car_insurance_model.impute_mean('Tax')

    ####################################################################################################################

    # one hot encoding certain attributes
    car_insurance_model.one_hot_encode_attribute('Marital_Status')  # one hot encodes Marital_Status

    ####################################################################################################################

    # found credit score to have an interesting value of 9999 for some customers, I attempted to one hot encode all the
    # customers that had this score to a new column however, found this to have no significant difference on the model
    # however, I decided to leave the code in the class DataPreprocessor:

    # car_insurance_model.new_column_infinite_credit_score()

    ####################################################################################################################

    # attempted to impute using knn from a package known as fancyimpute however, I found this to be extremely
    # inefficient and instead used standard methods. the code is left the class DataPreprocessor and called on the next
    # line:
    # car_insurance_model.impute_knn(3)

    ####################################################################################################################

    # Impute the other attributes using standard methods
    car_insurance_model.impute_median('Credit_Score')
    car_insurance_model.impute_mode('Veh_Mileage')
    car_insurance_model.impute_median('License_Length')  # should try to impute by first categorising by Maritial_Status
    car_insurance_model.impute_mode('Veh_Value')  # should find a better way
    car_insurance_model.impute_median('Age')

    ####################################################################################################################

    # check all values have been imputed
    print('After imputing all the attributes, the missing ratio is found to be:')
    car_insurance_model.missing_data_ratio_print()

    ####################################################################################################################
    # Transform the data

    # car_insurance_model.histogram_and_q_q('Price')
    # max_price = car_insurance_model._data_set['Price'].max()
    # car_insurance_model._data_set['Price'] = max_price + 1 - car_insurance_model._data_set['Price']
    # car_insurance_model.boxcox_trans_attribute('Price', 0.1)
    # car_insurance_model._data_set['Price'] = np.sqrt(car_insurance_model._data_set['Price'])
    # car_insurance_model.histogram_and_q_q('Price')

    ####################################################################################################################
    # MODEL
    ####################################################################################################################

    car_insurance_model.shuffle_data_set()  # shuffle the data set before splitting

    #   must split data to 75% training, 25% test with a seed of 2
    car_insurance_model.split_data_set_into_train_x_test_x_train_y_test_y('Sale', 0.25, 2)

    car_insurance_model.knn_model(5, 10)  # fit a knn with k=5 and print percentage accuracy for 10-fold cross
    # validation and confusion matrix against the test set

    # car_insurance_model.svm_model('auto', 10)  # fit a svm and print percentage accuracy for 10-fold cross and
    # confusion matrix against the test set


if __name__ == "__main__":
    main()
