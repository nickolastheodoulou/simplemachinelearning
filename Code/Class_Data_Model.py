import pandas as pd
from Code.Class_Data_Preprocessing import DataPreprocessing
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from scipy.special import inv_boxcox
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


class DataModel(DataPreprocessing):
    def __init__(self, train_x, test_x):
        super().__init__(train_x, test_x)

    # Create a function called lasso
    # Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
    def lasso_compare_alpha(self, alphas):
        df = pd.DataFrame()  # Create an empty data frame
        df['Feature Name'] = self._train_x.columns  # Create a column of feature names
        for alpha in alphas:  # For each alpha value in the list of alpha values,

            #  Robustscaler() makes the lasso more robust on outliers
            # Create a lasso regression with that alpha value,
            lasso = make_pipeline(RobustScaler(), sklearn.linear_model.Lasso(alpha=alpha, random_state=1))
            lasso.fit(self._train_x, self._train_y)  # Fit the lasso regression
            column_name = 'Alpha = %f' % alpha  # Create a column name for that alpha value
            df[column_name] = lasso.steps[1][1].coef_  # Create a column of coefficient values
        return df  # Return the data frame

    def linear(self, attribute):  # simple linear model
        regr = sklearn.linear_model.LinearRegression()  # Create linear regression object
        regr.fit(self._train_x, self._train_y)  # Train the model using the training sets
        pred_y_model = regr.predict(self._test_x)  # Make predictions using the testing set
        pred_y_model = inv_boxcox(pred_y_model, 0.1)  # inverse boxcox the prediction
        pred_y_model = pd.DataFrame(data=pred_y_model, columns={attribute})  #
        pred_y_model = pd.concat([self._test_y_id, pred_y_model], axis=1)
        return pred_y_model

    def lasso(self, alpha, attribute):
        lasso = make_pipeline(RobustScaler(), sklearn.linear_model.Lasso(alpha=alpha, random_state=1))
        lasso.fit(self._train_x, self._train_y)
        pred_y_model = lasso.predict(self._test_x)  # Make predictions using the testing set
        pred_y_model = inv_boxcox(pred_y_model, 0.1)  # inverse boxcox the prediction
        pred_y_model = pd.DataFrame(data=pred_y_model, columns={attribute})  #
        pred_y_model = pd.concat([self._test_y_id, pred_y_model], axis=1)
        return pred_y_model

    def svm(self):
        clf = SVC(gamma='auto')
        clf.fit(self._train_x, y=self._train_y)
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto',
            kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        pred_y_model = clf.predict(self._test_x)
        pred_y_model = pd.DataFrame(data=pred_y_model, columns={'classification'})
        return pred_y_model

    def neuralnetwork(self):  # implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 12, 5), random_state=1)
        clf.fit(self._train_x, self._train_y)
        pred_y_model = clf.predict(self._test_x)
        return pred_y_model
