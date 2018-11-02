import pandas as pd
from scipy.special import boxcox
import numpy as np
from Class_Data_Modeler import DataModeler
from Class_Data_Preprocessor import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

def main():
    ####################################################################################################################
    # Main used for house price data set

    #  at this point once the data has been explored, want train_Y to be in its own variable separate from train_X to
    #  pre-process the data train_X and test_X should not be combined at any point as the data should be preprocessed in
    #  one go for train_X but in a real world scenario, test_X may not come in as a large dataset

    model_adult = DataModeler(pd.read_csv("Data_In/Adult/adult.data.txt", header=None, sep=",\s", na_values=["?"])
                            , pd.read_csv("Data_In/Adult/adult.test.txt", header = None, sep=",\s", na_values=["?"]))
    model_adult._train_data_set.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                            "hours-per-week", "native-country", "salary"]
    print(model_adult._train_data_set)
    model_adult._test_data_set.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                            "hours-per-week", "native-country", "salary"]
    #model_adult.box_plot("age","salary")
    #model_adult.missing_data_ratio_bar_graph()
    #model_adult.scatter_plot("hours-per-week","age")
    #model_adult.train_missing_data_ratio_print()
    # model_adult.histogram_and_q_q("age")
    # model_adult.histogram_and_q_q("hours-per-week")
    # model_adult.histogram_and_q_q("capital-gain")
    # model_adult.histogram_and_q_q("capital-loss")
    #     # model_adult.bar_graph_attribute("occupation")
    #     # model_adult.bar_graph_attribute("workclass")

    # print(model_adult._train_data_set["capital-loss"])
    #
    # model_adult.normalise_attribute("capital-loss")
    # model_adult.box_cox_trans_attribute("capital-loss", 5)
    # model_adult.normalise_attribute("capital-loss")
    # print(model_adult._train_data_set["capital-loss"])
    # model_adult.histogram_and_q_q("capital-loss")
    model_adult.move_target_to_train_y("salary")

    #print(model_adult._x_train)
   # model_adult.random_forest()
    model_adult.drop_attribute("fnlwgt")
    model_adult.drop_attribute("workclass")
    model_adult.drop_attribute("education-num")
    model_adult.drop_attribute("marital-status")
    model_adult.drop_attribute("occupation")
    model_adult.one_hot_encode_attribute("relationship")
    model_adult.one_hot_encode_attribute("education")
    model_adult.drop_attribute("capital-gain")
    model_adult.drop_attribute('capital-loss')
    model_adult.drop_attribute("hours-per-week")
    model_adult.one_hot_encode_attribute("race")
    model_adult.one_hot_encode_attribute("sex")
    model_adult.drop_attribute("native-country")
    model_adult.drop_attribute('age')
    print(model_adult._test_data_set)

    #model_adult.random_forest()
    model_adult.shuffle_data_set()

    #print(model_adult._test_data_set)

    model_adult._y_test = model_adult._test_data_set["salary"]
    model_adult._test_data_set = model_adult._test_data_set.drop(["salary"], axis=1)

    model_adult.delete_unnecessary_one_hot_encoded_columns()

    for i in range(len(model_adult._y_test.values)):
        model_adult._y_test.values[i] = model_adult._y_test.values[i].strip('.')

    my_random_forest_model = model_adult.random_forest()

if __name__ == "__main__":
    main()
