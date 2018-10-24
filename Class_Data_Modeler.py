import pandas as pd

from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from Class_Data_Preprocessor import DataPreprocessor


class DataModeler(DataPreprocessor):
    def __init__(self, train_data_set, test_data_set):
        super().__init__(train_data_set, test_data_set)

    def knn_model_grid_search(self, tuned_parameters, number_of_folds):
        # calls teh function to perform the grid search for user inputted parameters
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
        print('For k-NN when k=', number_of_neighbours, ' the percentage accuracy is',
              my_knn_model.score(self._x_test, self._y_test))

        # print confusion matrix
        print('The confusion matrix for k-NN when k=', number_of_neighbours, 'is: ',
              pd.crosstab(self._y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        # Applying K-Fold cross validation

        # can add n_jobs =-1 to set all CPU's to work
        percent_accuracies = cross_val_score(estimator=my_knn_model, X=self._x_train, y=self._y_train,
                                             cv=number_of_folds) * 100

        print('For k-NN when k=', number_of_neighbours, ' the percentage accuracy of each ', number_of_folds,
              '-fold is:', percent_accuracies)

    # method that performs a grid search for svm
    def svm_model_grid_search(self, tuned_parameters, number_of_folds):

        # calls teh function to perform the grid search for user inputted parameters
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

    # method that performs k-fold cross validation on an SVM model with user inputted parameters
    def svm_model(self, my_gamma, my_c, number_of_folds):
        # creates a SVM classifier
        my_svm_model = SVC(C=my_c, decision_function_shape='ovo', degree=3, gamma=my_gamma, kernel='rbf')
        my_svm_model.fit(self._x_train, self._y_train)  # fits the SVM model to sample data

        # C : Penalty parameter of the error term, default is 1.0
        # degree : Degree of the polynomial kernel function
        # gamma : Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        # kernel : specifies the kernel type used in the algorithm

        y_pred = my_svm_model.predict(self._x_test)  # set the predicted values to the prediction using x_test

        # print percent of correct predictions
        print('For SVM when gamma=auto, percentage accuracy is: ', my_svm_model.score(self._x_test,
                                                                                      self._y_test))
        # print confusion matrix

        print('The confusion matrix for the SVM when gamma=', my_gamma,
              pd.crosstab(self._y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        # can add n_jobs =-1 to set all CPU's to work
        percent_accuracies = cross_val_score(estimator=my_svm_model, X=self._x_train, y=self._y_train,
                                             cv=number_of_folds, n_jobs=-1) * 100

        print('For SVM when gamma=', my_gamma, ' the percentage accuracy of each ', number_of_folds, '-fold is:',
              percent_accuracies)

    # implements a multi-layer perceptron (MLP) algorithm that trains using Back-propagation
    def neural_network_model(self, my_alpha, my_hidden_layers, number_of_folds):
        my_neural_network_model = MLPClassifier(solver='lbfgs', alpha=my_alpha, hidden_layer_sizes=my_hidden_layers,
                                                random_state=1)

        my_neural_network_model.fit(self._x_train, self._y_train)
        y_pred = my_neural_network_model.predict(self._x_test)

        # print percent of correct predictions
        print('For SVM when gamma=auto, percentage accuracy is: ', my_neural_network_model.score(self._x_test,
                                                                                                 self._y_test))
        # print confusion matrix

        print('The confusion matrix for the neural_network_model when alpha=', my_alpha,
              ' and the hidden layers being set to: ', my_hidden_layers,
              pd.crosstab(self._y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

        # can add n_jobs =-1 to set all CPU's to work
        percent_accuracies = cross_val_score(estimator=my_neural_network_model, X=self._x_train, y=self._y_train,
                                             cv=number_of_folds, n_jobs=-1) * 100

        print('For the neural_network_model when alpha=', my_alpha, ' and the hidden layers being set to: ',
              my_hidden_layers, ' the percentage accuracy of each ', number_of_folds, '-fold is:', percent_accuracies)
