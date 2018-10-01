import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from Class_Data_Model import DataModel


def main():
    data = pd.read_csv('Data_In/iris.txt', header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'classification']

    train = data.iloc[:70, :]
    test = data.iloc[70:, :]
    model_df = DataModel(train, test)

    #print(model_df._train_X.head())
    #print(model_df._test_X.head())
    #model_df.describe_attribute('sepal_length')
    #model_df.histogram_and_q_q('sepal_length')

    model_df.move_target_to_train_y('classification')
    model_df.move_target_to_test_y('classification')

    print(confusion_matrix(model_df._test_Y, model_df.SVM()))
    print(confusion_matrix(model_df._test_Y, model_df.neuralnetwork()))



if __name__ == "__main__":
    main()
