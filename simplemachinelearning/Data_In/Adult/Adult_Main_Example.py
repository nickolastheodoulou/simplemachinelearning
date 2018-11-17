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

    model_adult = DataModeler(pd.read_csv("Data_In/Adult/adult.data.txt", header=None, sep=",\s", na_values=["?"]),
                              pd.read_csv("Data_In/Adult/adult.test.txt", header=None, sep=",\s", na_values=["?"]))
    model_adult._train_data_set.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                           "hours-per-week", "native-country", "salary"]
    print(model_adult._train_data_set)
    model_adult._test_data_set.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                          "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                                          "hours-per-week", "native-country", "salary"]

    # model_adult.box_plot("age", "salary")
    # model_adult.missing_data_ratio_bar_graph()
    # model_adult.scatter_plot("hours-per-week", "age")
    model_adult.train_missing_data_ratio_print()
    model_adult.test_missing_data_ratio_print()

    # model_adult.histogram_and_q_q("age")
    # model_adult.box_cox_trans_attribute("age", 0.25)
    model_adult.normalise_attribute("age")
    # model_adult.histogram_and_q_q("age")

    model_adult.box_cox_trans_attribute("hours-per-week", 0.85)
    model_adult.normalise_attribute("hours-per-week")

    model_adult.box_cox_trans_attribute("capital-gain", 0.01)
    model_adult.normalise_attribute("capital-gain")

    model_adult.box_cox_trans_attribute("capital-loss", 0.01)
    model_adult.normalise_attribute("capital-gain")

    model_adult.box_cox_trans_attribute("fnlwgt", 0.45)
    model_adult.normalise_attribute("fnlwgt")

    # model_adult.histogram_and_q_q("capital-gain")
    # model_adult.histogram_and_q_q("capital-loss")
    # model_adult.bar_graph_attribute("occupation")
    # model_adult.bar_graph_attribute("workclass")

    # print(model_adult._train_data_set["capital-loss"])

    # model_adult.normalise_attribute("capital-loss")
    # model_adult.box_cox_trans_attribute("capital-loss", 5)
    # model_adult.normalise_attribute("capital-loss")
    # print(model_adult._train_data_set["capital-loss"])
    # model_adult.histogram_and_q_q("capital-loss")
    model_adult.shuffle_data_set()
    model_adult.move_target_to_train_y("salary")
    model_adult.move_target_to_test_y('salary')

    # print(model_adult._x_train)
    # model_adult.random_forest()

    # model_adult.box_cox_trans_attribute("age",1)
    # model_adult.normalise_attribute("age")
    # #model_adult.drop_attribute("hours-per-week")
    # model_adult.box_cox_trans_attribute("capital-gain",1)
    # model_adult.normalise_attribute("age")
    # #model_adult.box_cox_trans_attribute("capital-loss",1)
    # #model_adult.normalise_attribute("capital-loss")
    #
    #

    # model_adult.drop_attribute("fnlwgt")
    model_adult.one_hot_encode_attribute("workclass")
    model_adult.drop_attribute("education-num")
    model_adult.one_hot_encode_attribute("marital-status")
    model_adult.one_hot_encode_attribute("occupation")
    model_adult.drop_attribute("relationship")
    model_adult.one_hot_encode_attribute("education")
    # #model_adult.drop_attribute("capital-gain")
    # #model_adult.drop_attribute('capital-loss')
    # #model_adult.drop_attribute("hours-per-week")
    model_adult.one_hot_encode_attribute("race")
    model_adult.one_hot_encode_attribute("sex")
    model_adult.drop_attribute("native-country")
    # # #model_adult.drop_attribute('age')
    # print(model_adult._test_data_set)

    # #model_adult.random_forest()

    # #print(model_adult._test_data_set)
    # model_adult.delete_unnecessary_one_hot_encoded_columns()

    # delete the full stop in the test data set so that the test and predicted values can be compared
    for i in range(len(model_adult._y_test.values)):
        model_adult._y_test.values[i] = model_adult._y_test.values[i].strip('.')

    # eta typical final values to be used: 0.01-0.2
    # max_depth Typical values: 3 - 10
    # subsample Typical values: 0.5-1
    # colsample_bytree Typical values: 0.5-1
    # lambda
    # alpha

    # grid_param_xgboost = {'eta': [0.01, 0.1, 0.2], 'max_depth': [3, 6], 'gamma': [0, 0.1, 1],
    #                        "colsample_bytree": [0.5, 1]}
    # grid_param_xgboost = {'eta': [0.01], 'max_depth': [6], 'gamma': [0], "colsample_bytree": [1], 'lambda': [0, 0.1, 1],
    #                      'alpha': [0, 0.1, 1]}
    # grid_param_xgboost = {'eta': [0.01, 0.1, 0.2], 'max_depth': [6, 10]}
    # model_adult.classification_model_grid_search(XGBClassifier, grid_param_xgboost, 3)

    tuned_parameters_xgboost = {'eta': 0.01, 'min_child_weight': 1, 'max_depth': 6, 'gamma': 0, 'subsample': 1,
                                "colsample_bytree": 1, 'lambda': 0, 'alpha': 0}
    # tuned_parameters_xgboost = {'eta': 0.01, 'max_depth': 6}
    model_adult.classification_model(XGBClassifier, tuned_parameters_xgboost, 10)

    ####################################################################################################################
    # random forest model
    # tuned_parameters_random_forest = {'oob_score': True, 'n_estimators': 10, 'class_weight': {'>50K': 3.2, '<=50K': 1}}
    # model_adult.classification_model(RandomForestClassifier, tuned_parameters_random_forest, 10)
    ####################################################################################################################

    ####################################################################################################################
    # svm model
    # tuned_parameters_svm = {'C': 1, 'class_weight':{'>50K':3.2, '<=50K':1}, 'random_state':0}
    # my_svm = model_adult.classification_model(LinearSVC, tuned_parameters_knn, 10)
    ####################################################################################################################

    ####################################################################################################################
    # knn model
    # grid_param_knn ={'n_neighbors': [5, 7, 9]}
    # model_adult.classification_model_grid_search(KNeighborsClassifier, grid_param_knn, 3)
    #
    # tuned_parameters_knn = {'n_neighbors': 9}
    # my_knn = model_adult.classification_model(KNeighborsClassifier, tuned_parameters_knn, 10)
    ####################################################################################################################

    # need to return pred_y for models and create a method to submitt such as:
    '''
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    # submission.to_csv('../output/submission.csv', index=False)
    '''


if __name__ == "__main__":
    main()
