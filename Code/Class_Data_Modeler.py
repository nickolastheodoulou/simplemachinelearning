from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

from Code.Class_Data_Preprocessor import DataPreprocessor


class DataModeler(DataPreprocessor):
    def __init__(self, data_set):
        super().__init__(data_set)

    def knn_model(self, target):
        # set attributes to all other columns in the data_set
        attribute_matrix = self._data_set.loc[:, self._data_set.columns != target]

        #  split the data 50:50 between test and train
        x_train, x_test, y_train, y_test = train_test_split(attribute_matrix, self._data_set[target], test_size=0.50,
                                                            random_state=42)

        knn = neighbors.KNeighborsClassifier(n_neighbors=5)  # create a knn classifier with n=5
        knn_model_1 = knn.fit(x_train, y_train)  # fit the model to the data

        # define the predicted value of y and true value of y to create a prediction matrix
        y_true, y_pred = y_test, knn_model_1.predict(x_test)

        print('k-NN accuracy for test set: %f' % knn_model_1.score(x_test,
                                                                   y_test))  # print percent of correct predictions
        print(confusion_matrix(y_true, y_pred))  # print confusion matrix