from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pandas as pd
from sklearn.svm import SVC

from Class_Data_Preprocessor import DataPreprocessor


class DataModeler(DataPreprocessor):
    def __init__(self, data_set):
        super().__init__(data_set)
        self._x_train = 0
        self._x_test = 0
        self._y_train = 0
        self._y_test = 0

    def split_data_set_into_train_x_test_x_train_y_test_y(self, target, my_test_size, seed):
        # set attributes to all other columns in the data_set
        attribute_matrix = self._data_set.loc[:, self._data_set.columns != target]
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(attribute_matrix,
                                                                                    self._data_set[target],
                                                                                    test_size=my_test_size,
                                                                                    random_state=seed)

    def knn_model(self, my_number_of_neighbours):
        # create a knn classifier with n = my_number_of_neighbours
        my_knn_model = neighbors.KNeighborsClassifier(n_neighbors=my_number_of_neighbours)
        my_knn_model.fit(self._x_train, self._y_train)  # fit the knn classifier to the data

        # define the predicted value of y and true value of y to create a prediction matrix
        y_pred = my_knn_model.predict(self._x_test)

        # print percent of correct predictions
        print('k-NN accuracy for test set: %f' % my_knn_model.score(self._x_test, self._y_test))
        # print confusion matrix
        print(pd.crosstab(self._y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    def svm_model(self, my_gamma):
        my_svm_model = SVC(gamma=my_gamma)   # creates a SVM classifier
        my_svm_model.fit(self._x_train, self._y_train)  # fits the SVM model to sample data

        # C : Penalty parameter of the error term, default is 1.0
        # cache_size : Specify the size of the kernel cache (in MB).
        # class_weight : Set the parameter C of class i to class_weight[i]*C for SVC. default: all classes weight = 1.
        # coef0 : Independent term in kernel function
        # decision_function_shape : returns one-vs-one decision shape
        # degree : Degree of the polynomial kernel function
        # gamma : Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        # kernel : specifies the kernel type used in the algorithm
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3,
            gamma=my_gamma, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001,
            verbose=False)

        y_pred = my_svm_model.predict(self._x_test)  # set the predicted values to the prediction using x_test

        # print percent of correct predictions
        print('svm accuracy for test set: %f' % my_svm_model.score(self._x_test, self._y_test))
        # print confusion matrix
        print(pd.crosstab(self._y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))