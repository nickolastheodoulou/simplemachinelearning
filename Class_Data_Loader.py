import pandas as pd
import fancyimpute as fi
import sklearn
import matplotlib.pyplot as plt

class data_loader:  # class that creates the data matrix by initializing test_X and train_X from main
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




