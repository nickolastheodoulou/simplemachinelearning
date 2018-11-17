import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from Class_Data_Modeler import DataModeler


def main():
    # constructs the object car_insurance_model by loading in the data to the class DataModeler
    car_insurance_model = DataModeler(pd.read_csv("Data_In/Car_Insurance/DS_Assessment.csv"),
                                      pd.DataFrame({'null_df': []}))

    ####################################################################################################################
    # EXPLORATION
    ####################################################################################################################

    # Get a very broad understanding of the data from the dimension and first 5 rows

    # Prints the dimension of the data after successfully storing the data set as a Pandas data frame
    print("The dimension of the car insurance data is: ", car_insurance_model._train_data_set.shape)
    print(car_insurance_model._train_data_set.head())  # prints the first 5 rows of the data set

    ####################################################################################################################
    # Example of some of the graphs used to explore the data for the attribute: Age
    # These methods can and were used for all the attributes
    '''
    # counts number of each age
    car_insurance_model.attribute_value_count('Age')
    # counts Sale or NoSale for each number of age
    car_insurance_model.attribute_value_count_by_classification('Age', 'Sale')
    # displays and saves a bar graph showing the percentage of each value for the attribute Age in the data set
    car_insurance_model.bar_graph_attribute('Age')
    # displays a stacked Sale and NoSale bar graph for each attribute in Age in the data set
    car_insurance_model.bar_graph_attribute_by_classification('Age', 'Sale')
    # prints a summary of the distribution of the column 'Age' such as mean, standard deviation etc
    car_insurance_model.describe_attribute('Age')
    # plots a histogram of the attribute Age and also a quantile quantile plot
    # car_insurance_model.histogram_and_q_q('Age')
    # plots a scatter plot of Age and Price
    car_insurance_model.scatter_plot('Age', 'Price')
    # plots a scatter plot of Age and Price for Sale and NoSale
    car_insurance_model.scatter_plot_by_classification('Age', 'Price', 'Sale')

    car_insurance_model.histogram_and_q_q('Credit_Score')

    ####################################################################################################################
    # Observe how much data is missing for each attribute
    car_insurance_model.train_missing_data_ratio_print()
    # displays and saves a bar graph of the percentage of missing values
    car_insurance_model.missing_data_ratio_bar_graph()

    car_insurance_model.heat_map()
    '''

    ####################################################################################################################
    # PROCESSING
    ####################################################################################################################
    # Attempted to log and sqrt transform some skewed parameters however, I found the models to perform worse hence I
    # decided to instead normalise the attributes to have a mean of 0 and standard deviation of 1. Code below
    # demonstrates some of my attempts to better fit the data to a normal distribution

    '''
    car_insurance_model.histogram_and_q_q('Credit_Score')
    # max_price = car_insurance_model._data_set['Price'].max()
    # car_insurance_model._data_set['Price'] = max_price + 1 - car_insurance_model._data_set['Price']
    car_insurance_model.boxcox_trans_attribute('Credit_Score', 0.1)
    # car_insurance_model._data_set['Price'] = np.sqrt(car_insurance_model._data_set['Price'])
    car_insurance_model.histogram_and_q_q('Credit_Score')
    '''

    ####################################################################################################################

    # Normalise attributes to a mean of zero and standard deviation of 1 before imputing
    attributes_to_normalise = ['Veh_Mileage', 'Credit_Score', 'License_Length', 'Veh_Value', 'Price', 'Age', 'Tax']

    for i in attributes_to_normalise:
        car_insurance_model.normalise_attribute(i)

    ####################################################################################################################
    # creating new features from the attribute date

    # decided to add day_of_the_week column to see if any information can be extracted
    ####################################################################################################################
    # convert type of column date form object to datetime64
    car_insurance_model._train_data_set['Date'] = pd.to_datetime(car_insurance_model._train_data_set['Date'],
                                                                 infer_datetime_format=True)
    # add new column named days_of_the_week that has the day of the week
    car_insurance_model._train_data_set = car_insurance_model._train_data_set.assign(
        days_of_the_week=car_insurance_model._train_data_set['Date'].dt.weekday_name)
    ####################################################################################################################
    # bar graph of new column to see if any new information can be obtained
    # car_insurance_model.bar_graph_attribute_by_classification('days_of_the_week', 'Sale')
    # can see that on Friday typically there are less sales hence decided to create new column

    # used similar method to extract month and year, found month would have added too many attributes when one hot
    # encoding and year to not have any significant difference between 2015 and 2016

    # one hot encodes the column days_of_the_week by adding 7 new attributes
    car_insurance_model.one_hot_encode_attribute('days_of_the_week')
    # drop date as there are so many different days
    car_insurance_model.drop_attribute('Date')

    ####################################################################################################################
    # Dealing with the attributes Tax and Price

    # scatter plot the two attributes as they appear very highly correlated and could be used to impute the data
    # car_insurance_model.scatter_plot_by_classification("Tax", "Price")
    # found that tax and price follow two linear equations using car_insurance_model.scatter_plot("Tax", "Price")
    # the cutoff between following either equation was when the tax was between a value of 32 to 35 which was found by
    # looking through the data set:
    # typically when tax < 34, tax = 0.05 * price and when tax > 34, tax = 0.1 * price
    # hence this can be used to impute missing values more accurately

    # compare how many values are imputed using this method
    car_insurance_model.train_missing_data_ratio_print()
    ####################################################################################################################
    # impute price
    # as only 5 values are missing for both Price and Tax, the mean is imputed for these values

    # create a list of the index of the missing values in the price attribute
    price_missing_value_index = car_insurance_model._train_data_set[
        car_insurance_model._train_data_set['Price'].isnull()].index.tolist()

    # loop through the missing value index for price
    for i in price_missing_value_index:
        # if the value for tax in the same column as price is greater than 34
        if car_insurance_model._train_data_set['Tax'].values[i] > 33:
            # set the value for price equal to ten times the value of tax in the same column
            car_insurance_model._train_data_set['Price'].values[i] = car_insurance_model._train_data_set['Tax'].values[
                                                                         i] * 10
        else:
            # else set the price to 5 times the tax
            car_insurance_model._train_data_set['Price'].values[i] = car_insurance_model._train_data_set['Tax'].values[
                                                                         i] * 5

    print('The number of price values imputed is ', len(price_missing_value_index))
    ####################################################################################################################
    # impute tax
    # create a list of the index of the missing values in the tax attribute
    tax_missing_value_index = car_insurance_model._train_data_set[car_insurance_model._train_data_set['Tax'].isnull()].\
        index.tolist()
    # loop through the missing value index for price
    for i in tax_missing_value_index:
        # if the value for tax in the same column as price is greater than 34
        if car_insurance_model._train_data_set['Price'].values[i] > 330:
            # set the value for price equal to ten times the value of tax in the same column
            car_insurance_model._train_data_set['Tax'].values[i] = car_insurance_model._train_data_set['Price'].values[
                                                                       i] * 0.1
        else:
            # else set the price to 5 times the tax
            car_insurance_model._train_data_set['Tax'].values[i] = car_insurance_model._train_data_set['Price'].values[
                                                                       i] * 0.05

    print('The number of tax values imputed is ', len(tax_missing_value_index))
    ####################################################################################################################
    car_insurance_model.train_missing_data_ratio_print()

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

    # create a list of the indices of the credit score with a score of 9999
    credit_score_9999_index = car_insurance_model._train_data_set[car_insurance_model._train_data_set['Credit_Score'] ==
                                                                  9999].index.tolist()

    # create a new column for the credit scores with 9999 so they can be put into a different attribute
    car_insurance_model._train_data_set['Infinite_Credit_Score'] = 0

    for i in credit_score_9999_index:
        # set the new column values to 1 (one hot encoding)
        car_insurance_model._train_data_set['Infinite_Credit_Score'].values[i] = 1
        # drop the credit score of 9999 from the attribute Credit_Score
        car_insurance_model._train_data_set['Credit_Score'].values[i] = 0

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
    car_insurance_model.impute_mode('Veh_Value')  # should use a better method
    car_insurance_model.impute_median('Age')

    ####################################################################################################################

    # check all values have been imputed
    print('After imputing all the attributes, the missing ratio is found to be:')
    car_insurance_model.train_missing_data_ratio_print()

    ####################################################################################################################
    # MODELS
    ####################################################################################################################

    car_insurance_model.shuffle_data_set()  # shuffle the data set before splitting

    # split data to 75% training, 25% test with a seed set to 2 (to get the same split when running the code
    car_insurance_model.split_data_set_if_test_not_split('Sale', 0.25, 2)

    ####################################################################################################################
    # Knn model
    # gridsearch for knn

    # uncomment to run grid search
    grid_parameters_knn = [{'n_neighbors': [5, 15, 19]}]
    car_insurance_model.classification_model_grid_search(KNeighborsClassifier, grid_parameters_knn, 2)

    # fit a knn with k=5 and print percentage accuracy for 10-fold cross validation and confusion matrix against the
    # test set
    tuned_parameters_knn = {'n_neighbors': 19}
    car_insurance_model.classification_model(KNeighborsClassifier, tuned_parameters_knn, 10)

    ####################################################################################################################
    # SMV model
    # found these set of parameters to be the most optimum when performing a grid search

    # uncomment to run grid search
    tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [1/15, 1/16, 1/17], 'C': [11, 10, 12],
                            'decision_function_shape':['ovo']}]
    car_insurance_model.classification_model_grid_search(SVC, tuned_parameters_svm, 2)

    # fit a svm and print percentage accuracy for 10-fold cross and shows the confusion matrix for the best
    # hyper-parameters found when performing the grid-search

    # k-fold cross validation for optimum hyper-parameters to validate SVM model
    tuned_parameters_svm = {'C': 10, 'decision_function_shape': 'ovo', 'degree': 3, 'gamma': 1/16, 'kernel': 'rbf'}
    car_insurance_model.classification_model(SVC, tuned_parameters_svm, 10)
    ####################################################################################################################


if __name__ == "__main__":
    main()
