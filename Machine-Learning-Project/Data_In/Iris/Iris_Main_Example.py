import pandas as pd

from sklearn.svm import SVC

from Class_Data_Modeler import DataModeler


def main():
    ####################################################################################################################
    # Main used for iris data set
    iris_data = pd.read_csv('Data_In/Iris/iris.txt', header=None)
    iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'classification']

    model_iris = DataModeler(iris_data, 0)
    model_iris.split_data_set_if_test_not_split('classification', 0.7, 0)

    model_iris.heat_map()
    print(model_iris._train_data_set.head())
    print(model_iris._test_data_set.head())

    model_iris.describe_attribute('sepal_length')
    model_iris.histogram_and_q_q('sepal_length')

    tuned_parameters_svm = {'kernel': 'rbf', 'gamma': 1, 'C': 10, 'decision_function_shape': 'ovo'}
    model_iris.classification_model(SVC, tuned_parameters_svm, 10)

    ####################################################################################################################


if __name__ == "__main__":
    main()
