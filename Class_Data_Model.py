import pandas as pd
import matplotlib.pyplot as plt
from Class_Data_Loader import data_loader
from Class_Data_Preprocessing import data_preprocessing
import sklearn


class data_model(data_preprocessing):
    def __init__(self, test_X, train_X):
        super().__init__(test_X, train_X)
        self._Pred_Y = 0  # prediction object

    # Create a function called lasso,
    def lasso(self, alphas):  # Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
        df = pd.DataFrame()  # Create an empty data frame
        df['Feature Name'] = self._train_X.columns  # Create a column of feature names
        for alpha in alphas:  # For each alpha value in the list of alpha values,
            lasso = sklearn.linear_model.Lasso(alpha=alpha)  # Create a lasso regression with that alpha value,
            lasso.fit(self._train_X, self._train_Y)  # Fit the lasso regression
            column_name = 'Alpha = %f' % alpha  # Create a column name for that alpha value
            df[column_name] = lasso.coef_  # Create a column of coefficient values
        return df  # Return the dataframe

    def linear(self):# simple linear model
        regr = sklearn.linear_model.LinearRegression()# Create linear regression object
        regr.fit(self._train_X, self._train_Y)# Train the model using the training sets
        Pred_Y_list = regr.predict(self._test_X)# Make predictions using the testing set
        Pred_Y = pd.DataFrame(data=Pred_Y_list, columns={'SalePrice'})#
        _pred_Y_with_index = pd.concat([self._pred_Y, Pred_Y], axis=1)
        return _pred_Y_with_index
