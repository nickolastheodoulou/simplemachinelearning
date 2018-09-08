import pandas as pd
import fancyimpute as fi
import sklearn
import matplotlib.pyplot as plt

class data_matrix:  # class that creates the data matrix by initializing test_X and train_X from main
    def __init__(self, test_X, train_X):#initialise the object with the test and train matrices from the CSV file
        self._test_X = test_X                           #test dataframe
        self._train_X = train_X                         #train dataframe
        self._id = 0                                    #id column later used to add to Pred_Y
        self._label_train_Y = "SalePrice"               #label of train_Y used for graphs
        self._train_Y = 0                               #train_Y dataframe(data later loaded in)
        self._test_X_dim = 0                            #dimension of test_X
        self._train_X_dim = 0                           #dimension of train_X

        self._train_X_string = 0                        #all string attributes of train_X
        self._test_X_string = 0                         #all string attributes for test_X
        self._train_X_int_float = 0                     #all int and float attributes for train_X
        self._test_X_int_float = 0                      #all int and float attributes for test_X


        self._Pred_Y = 0                                #prediction object
        # the underscore means that the members are protected



    def dim_data(self):#function that finds the dimensions of both the train and test set and stores them in the object
        self._train_X_dim = self._train_X.shape
        self._test_X_dim = self._test_X.shape
        return None

    def first_column_drop(self):  # method that drops the first column of both train_X and test_X
        self._train_X = self._train_X.drop(self._train_X.columns[0], axis=1)#drops the first column of the train set as the id so that it isnt included in the model
        self._id = self._test_X['Id']#define id so that it can be added to pred_Y
        self._test_X = self._test_X.drop(self._test_X.columns[0], axis=1)  # drops the first column of the test set as the id so that it isnt included in the model
        return None

    def move_classification_to_train_y(self):#function that removes the last column of Train_X and puts it into Train_Y
        final_column = self._train_X_dim[1] - 1#finds the final column using the train_X_dim variable
        self._train_Y = self._train_X.iloc[:, final_column]#updates the dataframe train_Y
        self._train_X = self._train_X.drop(self._train_X.columns[(self._train_X.shape[1]-1)], axis=1)  # drops the first column of the train set as it has been moved
        return None

    def split_attributes(self):#method that updates the variables: _train_X_string, _test_X_string, _train_X_int_float, _test_X_int_float
        self._train_X_string = self._train_X.select_dtypes(include=['object']).copy()#updates the dataset variable: _train_X_string that contains all the "object" datatypes in train
        self._test_X_string = self._test_X.select_dtypes(include=['object']).copy()  # updates the dataset variable: _test_X_string that contains all the "object" datatypes in test
        self._train_X_int_float = self._train_X.select_dtypes(include=['int64', 'float64']).copy()# updates the dataset variable: _train_X_string that contains all the "object" datatypes in train
        self._test_X_int_float = self._test_X.select_dtypes(include=['int64', 'float64']).copy()# updates the dataset variable: _test_X_string that contains all the "object" datatypes in test
        return None

    def normalise_data(self):#normalises the desired dataframes
        self._train_X_int_float = (self._train_X_int_float - self._train_X_int_float.mean()) / self._train_X_int_float.std()#normalise _train_X_int_float using standard score
        self._test_X_int_float = (self._test_X_int_float - self._test_X_int_float.mean()) / self._test_X_int_float.std()
        #_train_X_string and _test_X_string are not normalised as there is no point when one hot encoding
        #also shouldnt normalise train_Y

    def one_hot_encoding(self):#function that performs one_hot_encoding on the variables _train_X_string and _test_X_string then updates the variables
        self._train_X_string = pd.get_dummies(self._train_X_string)  # method to convert all the string attributes into one hot encoded by updating the dataframe from pandas
        self._test_X_string = pd.get_dummies(self._test_X_string)  # method to convert all the string attributes into one hot encoded
        return None

    def fill_missing_values(self):#function that inputs the missing values into _train_X_int_float
        X_filled_knn = fi.KNN(k=3).complete(self._train_X_int_float)  # completes the missing attributes using KNN from fancy impute using the 3 closes complete columns
        self._train_X_int_float = pd.DataFrame(X_filled_knn, columns=self._train_X_int_float.columns.copy())  #updates _train_X_int_float with the missing data

####################################################################################################################################################################################
# using knn for test data to quickly fill in missing data to see if i can build a model(later will input with a mean from train or something else
#
        X_filled_knn_test = fi.KNN(k=3).complete(self._test_X_int_float)  # completes the missing attributes using KNN from fancy impute using the 3 closes complete columns
        self._test_X_int_float = pd.DataFrame(X_filled_knn_test, columns=self._test_X_int_float.columns.copy())  # updates _train_X_int_float with the missing data
#
####################################################################################################################################################################################

    def combine_string_int_float(self):#first finds the missing columns of test_X and fills them with zeros in the correct place
        #might run into error if test_X has more columns than train_X in other situations
        #combines the int/float matrices and the string matrices and update them to _train_X and _test_X

        missing_cols = set(self._train_X_string.columns) - set(self._test_X_string.columns)  # Get missing columns in the training test
        for i in missing_cols:  # Add a missing column in test set with default value equal to 0
            self._test_X_string[i] = 0
        self._test_X_string = self._test_X_string[self._train_X_string.columns]  # Ensure the order of column in the test set is in the same order than in train set

        self._train_X = pd.concat([self._train_X_int_float, self._train_X_string], axis=1)#combines the train_X
        self._test_X = pd.concat([self._test_X_int_float, self._test_X_string], axis=1)


    def export_CSV(self):#exports the cleaned train_X, train_Y and test_X data to a seperate CSV file
        self._train_X.to_csv('Data_Out/train_X_up.csv', index=False)
        self._test_X.to_csv('Data_Out/test_X_up.csv', index=False)
        self._train_Y.to_csv('Data_Out/test_Y_up.csv', index=False)


