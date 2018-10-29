import pandas as pd

from Class_Data_Modeler import DataModeler


def main():
    ####################################################################################################################
    # Main used for iris data set
    iris_data = pd.read_csv('Data_In/Iris/iris.txt', header=None)
    iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'classification']

    model_iris = DataModeler(iris_data, 0)
    model_iris.split_data_set_if_test_not_split('classification', 0.7, 0)

    model_iris.heat_map()
    print(model_iris._x_train.head())
    print(model_iris._x_test.head())

    model_iris.describe_attribute('sepal_length')
    model_iris.histogram_and_q_q('sepal_length')

    model_iris.split_data_data_set_X_data_set_y('classification')

    model_iris.svm_model('auto', 1, 10)
    model_iris.neural_network_model(1e-5, (25, 12, 5), 10)

    ####################################################################################################################


if __name__ == "__main__":
    main()