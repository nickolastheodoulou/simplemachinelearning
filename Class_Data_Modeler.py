import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from Class_Data_Preprocessor import DataPreprocessor


class DataModeler(DataPreprocessor):
    def __init__(self, data_set):
        super().__init__(data_set)
        self._x_train = 0
        self._x_test = 0
        self._y_train = 0
        self._y_test = 0
        self._data_set_y = 0
        self._data_set_X = 0

    def split_data_set_into_train_x_test_x_train_y_test_y(self, target, my_test_size, seed):
        # set attributes to all other columns in the data_set
        attribute_matrix = self._data_set.loc[:, self._data_set.columns != target]
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(attribute_matrix,
                                                                                    self._data_set[target],
                                                                                    test_size=my_test_size,
                                                                                    random_state=seed)

    def knn_model_grid_search(self, tuned_parameters, number_of_folds):
        # calls teh function to perform the gridearch for usear inputted parameters
        my_knn_model = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters, cv=number_of_folds,
                                    scoring='f1_macro', n_jobs=-1)

        # fits the knn models to sample data
        my_knn_model.fit(self._x_train, self._y_train)

        y_true, y_pred = self._y_test, my_knn_model.predict(self._x_test)
        print(classification_report(y_true, y_pred))  # prints a summary of the grid search

        print('The best parameters for the model is', my_knn_model.best_params_)  # prints the best parameters found

        print('The results are:', my_knn_model.cv_results_)  # prints all the results

        # prints the scoring for each model in the grid
        for param, score in zip(my_knn_model.cv_results_['params'], my_knn_model.cv_results_['mean_test_score']):
            print(param, score)

    def knn_model(self, number_of_neighbours, number_of_folds):
        # create a knn classifier with n = my_number_of_neighbours
        my_knn_model = neighbors.KNeighborsClassifier(n_neighbors=number_of_neighbours)

        my_knn_model.fit(self._x_train, self._y_train)  # fit the knn classifier to the data

        # define the predicted value of y and true value of y to create a prediction matrix
        y_pred = my_knn_model.predict(self._x_test)

        # print percent of correct predictions
        print('For k-NN when k=', number_of_neighbours, ' the percentage accuracy is', my_knn_model.score(self._x_test,
                                                                                                          self._y_test))
        # print confusion matrix
        print('The confusion matrix for k-NN when k=', number_of_neighbours, 'is: ',
              pd.crosstab(self._y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        # Applying K-Fold cross validation

        # can add n_jobs =-1 to set all cpus to work
        percent_accuracies = cross_val_score(estimator=my_knn_model, X=self._x_train, y=self._y_train,
                                             cv=number_of_folds) * 100

        print('For k-NN when k=', number_of_neighbours, ' the percentage accuracy of each ', number_of_folds,
              '-fold is:', percent_accuracies)

    # method that performs a grid search for svm
    def svm_model_grid_search(self, tuned_parameters, number_of_folds):

        # calls teh function to perform the gridearch for usear inputted parameters
        my_svm_model = GridSearchCV(SVC(decision_function_shape='ovo', degree=3), tuned_parameters, cv=number_of_folds,
                                    scoring='f1_macro', n_jobs=-1)

        my_svm_model.fit(self._x_train, self._y_train)  # fits the SVM models to sample data

        y_true, y_pred = self._y_test, my_svm_model.predict(self._x_test)
        print(classification_report(y_true, y_pred))  # prints a summary of the grid search

        print('The best parameters for the model is', my_svm_model.best_params_)  # prints the best parameters found

        print('The results are:', my_svm_model.cv_results_)  # prints all the results

        # prints the scoring for each model in the grid
        for param, score in zip(my_svm_model.cv_results_['params'], my_svm_model.cv_results_['mean_test_score']):
            print(param, score)

    # method that peforms k-fold cross validation on an SVM model with user inputted parameters
    def svm_model(self, my_gamma, my_c, number_of_folds):
        my_svm_model = SVC(gamma=my_gamma)  # creates a SVM classifier
        my_svm_model.fit(self._x_train, self._y_train)  # fits the SVM model to sample data

        # C : Penalty parameter of the error term, default is 1.0
        # degree : Degree of the polynomial kernel function
        # gamma : Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        # kernel : specifies the kernel type used in the algorithm
        SVC(C=my_c, decision_function_shape='ovo', degree=3, gamma=my_gamma, kernel='rbf')

        y_pred = my_svm_model.predict(self._x_test)  # set the predicted values to the prediction using x_test

        # print percent of correct predictions
        print('For SVM when gamma=auto, percentage accuracy is: ', my_svm_model.score(self._x_test, self._y_test))
        # print confusion matrix
        print('The confusion matrix for the SVM when gamma=auto is: ',
              pd.crosstab(self._y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        # can add n_jobs =-1 to set all CPU's to work
        percent_accuracies = cross_val_score(estimator=my_svm_model, X=self._x_train, y=self._y_train,
                                             cv=number_of_folds, n_jobs=-1) * 100

        print('For SVM when gamma=', my_gamma, ' the percentage accuracy of each ', number_of_folds, '-fold is:',
              percent_accuracies)

