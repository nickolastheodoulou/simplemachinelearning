import pandas as pd

from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from scipy.special import inv_boxcox

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

    # Create a function called lasso
    # Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
    def lasso_compare_alpha(self, alphas):
        df = pd.DataFrame()  # Create an empty data frame
        df['Feature Name'] = self._train_data_set.columns  # Create a column of feature names
        for alpha in alphas:  # For each alpha value in the list of alpha values,
            #  Robustscaler() makes the lasso more robust on outliers
            # Create a lasso regression with that alpha value,
            lasso = make_pipeline(RobustScaler(), Lasso(alpha=alpha, random_state=1))
            lasso.fit(self._train_data_set, self._y_train)  # Fit the lasso regression
            column_name = 'Alpha = %f' % alpha  # Create a column name for that alpha value
            df[column_name] = lasso.steps[1][1].coef_  # Create a column of coefficient values
        return df  # Return the data frame

    def lasso_model(self, alpha, attribute):
        my_lasso_model = make_pipeline(RobustScaler(), Lasso(alpha=alpha, random_state=1))
        my_lasso_model.fit(self._train_data_set, self._y_train)
        y_pred = my_lasso_model.predict(self._test_data_set)  # Make predictions using the testing set
        # pred_y_model = inv_boxcox(pred_y_model, 0.1)  # inverse boxcox the prediction

        # export predictions as csv
        y_pred = pd.DataFrame(data=y_pred, columns={attribute})  #
        y_pred = pd.concat([self._test_y_id, y_pred], axis=1)
        y_pred.to_csv('Data_Out/Lasso_Model_alpha_' + str(alpha) + ' _for_ ' + attribute + '.csv', index=False)

        # print cross validation scores
        scores = cross_validate(my_lasso_model, self._train_data_set, self._y_train, cv=10,
                                scoring=('explained_variance', 'neg_mean_absolute_error', 'r2',
                                         'neg_mean_squared_error'))

        # print the scores for test
        print('For LASSO, the explained_variance scores are: ', scores['test_explained_variance'])
        print('For LASSO, the neg_mean_absolute_error scores are: ', scores['test_neg_mean_absolute_error'])
        print('For LASSO, the neg_mean_squared_error scores are: ', scores['test_neg_mean_squared_error'])
        print('For LASSO, the R^2 scores are: ', scores['test_r2'])

    def linear_model(self, attribute):  # simple linear model
        my_linear_model = LinearRegression()  # Create linear regression object
        my_linear_model.fit(self._train_data_set, self._y_train)  # Train the model using the training sets
        pred_y_model = my_linear_model.predict(self._test_data_set)  # Make predictions using the testing set
        # pred_y_model = inv_boxcox(pred_y_model, 0.1)  # inverse boxcox the prediction
        pred_y_model = pd.DataFrame(data=pred_y_model, columns={attribute})  #
        pred_y_model = pd.concat([self._test_y_id, pred_y_model], axis=1)
        pred_y_model.to_csv('Data_Out/Linear_Model.csv', index=False)

        # print cross validation scores
        scores = cross_validate(my_linear_model, self._train_data_set, self._y_train, cv=10,
                                scoring=('explained_variance', 'neg_mean_absolute_error', 'r2',
                                         'neg_mean_squared_error'))

        # print the scores for test
        print('For linear model, the explained_variance scores are: ', scores['test_explained_variance'])
        print('For linear model, the neg_mean_absolute_error scores are: ', scores['test_neg_mean_absolute_error'])
        print('For linear model, the neg_mean_squared_error scores are: ', scores['test_neg_mean_squared_error'])
        print('For linear model, the R^2 scores are: ', scores['test_r2'])
