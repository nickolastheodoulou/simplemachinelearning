import pandas as pd
from Class_Data_Preprocessing import DataPreprocessing
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from scipy.special import boxcox, inv_boxcox, boxcox1p, inv_boxcox1p
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class DataModel(DataPreprocessing):
    def __init__(self, train_X, test_X):
        super().__init__(train_X, test_X)

    # Create a function called lasso,
    def lasso_compare_alpha(self, alphas):  # Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
        df = pd.DataFrame()  # Create an empty data frame
        df['Feature Name'] = self._train_X.columns  # Create a column of feature names
        for alpha in alphas:  # For each alpha value in the list of alpha values,

            #Robustscaler() makes the lasso more robust on outliers
            lasso = make_pipeline(RobustScaler(), sklearn.linear_model.Lasso(alpha=alpha, random_state=1))  # Create a lasso regression with that alpha value,
            lasso.fit(self._train_X, self._train_Y)  # Fit the lasso regression
            column_name = 'Alpha = %f' % alpha  # Create a column name for that alpha value
            df[column_name] = lasso.steps[1][1].coef_  # Create a column of coefficient values
        return df  # Return the dataframe

    def linear(self, attribute):# simple linear model
        regr = sklearn.linear_model.LinearRegression()  # Create linear regression object
        regr.fit(self._train_X, self._train_Y)  # Train the model using the training sets
        pred_Y_model = regr.predict(self._test_X)  # Make predictions using the testing set
        pred_Y_model = inv_boxcox(pred_Y_model, 0.1)  # inverse boxcox the prediction
        pred_Y_model = pd.DataFrame(data=pred_Y_model, columns={attribute})  #
        pred_Y_model = pd.concat([self._test_Y_id, pred_Y_model], axis=1)
        return pred_Y_model

    def lasso(self, alpha, attribute):
        lasso = make_pipeline(RobustScaler(), sklearn.linear_model.Lasso(alpha=alpha, random_state=1))
        lasso.fit(self._train_X, self._train_Y)
        pred_Y_model = lasso.predict(self._test_X)  # Make predictions using the testing set
        pred_Y_model = inv_boxcox(pred_Y_model, 0.1)  # inverse boxcox the prediction
        pred_Y_model = pd.DataFrame(data=pred_Y_model, columns={attribute})  #
        pred_Y_model = pd.concat([self._test_Y_id, pred_Y_model], axis=1)
        return pred_Y_model

    def SVM(self):
        clf = SVC(gamma='auto')
        clf.fit(self._train_X, y=self._train_Y)
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto',
            kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        pred_Y_model = clf.predict(self._test_X)
        pred_Y_model = pd.DataFrame(data=pred_Y_model, columns={'classification'})
        return pred_Y_model

    def neuralnetwork(self):  # implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 12, 5), random_state=1)
        clf.fit(self._train_X, self._train_Y)
        pred_Y_model = clf.predict(self._test_X)
        return pred_Y_model
