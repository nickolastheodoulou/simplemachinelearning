# Machine Learning Techniques: Predicting The Price Of Houses
 ###Outline of Project
This is a repository implementing some of the machine learning algorithms and techniques I have learnt to predict house prices using a
 data set from the Kaggle competition: 'House Prices: Advanced Regression Techniques' where details of the dataset can be found [here.](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 
 Object Oriented Programming implementing inheritance was used to structure the code in a concise manner in which the sale price of 1459 
 houses were predicted using a data set of 1460 houses along with 79 attributes. The code using a Lasso model with alpha = 0.01 and 
 box cox transformations for all the attributes where lambda = 0.1 scored 1973 out of 4056 submissions.
 
 ####Description of the file structure
 The Data_In folder contains 3 files: the train data in train.CSV, the test data in test.CSV and a description
 of the attributes in data_description.text. The final column in train.CSV contains the target attribute: 'SalePrice' 
 which is what was predicted on the test data using a Lasso model. The Data_Out folder contains
 output CSV files of the models with various parameters and Plots contains examples of some of the plots produced. Main.py 
 is where the other four classes 
 * DataLoader within Class_Data_Loader.py
 * DataExploration within Class_Data_Exploration.py
 * DataPreprocessing within Class_Data_Preprocessing.py
 * DataModel within Class_Data_Model.py
 
 are called and executed. Class_Data_Loader loads the data into a Pandas dataframe. Class_Data_Exploration 
 contains methods that produce plots such as scatter graphs, histograms and box and whisker diagrams. The 
 purpose of this class is to deal with outliers and missing data. The method missing_data_ratio_and_bar_graph()
 is useful to see the percentage of missing values for each attribute which can be dealt with accordingly using
 the class DataPreprocessing. The class DataPreprocessing modifies both the test and train data
 using methods that fill in the missing values for both data sets. The data is then transformed using a Box Cox 
 transformation and normalised. The class DataModel can then be used to implement a method such as linear 
 regression or a Lasso to return the predicted value of the target for each test sample.
 
 The code was written in such a way that it would be possible to make a quick model on any data in a 
 CSV file by first loading it into a pandas dataframe, explore the data, preprocess the data and then
 feed it into a model that will produce an accurate outcome. This was done by using inheritance where
DataModel -> DataPreprocessing -> DataExploration -> DataLoader and the arrow 
is a symbol indicating 'inherited from'. 

This was done so that if one only wanted to only explore the data, only
the functions within the class DataExploration DataLoader are available when an object is constructed using DataExploration.
If a model is to be fitted then DataModel can be called to create an object where none of the data hasn't been tampered with
If inheritance wasn't used then it would be more difficult to keep track of how the test and train data were modified. 

 Full documentation of the methods within the classes will be added at a later date.

##Packages Required:
Environment set up in python 3.5 and runs with following packages:
* numpy        V:1.14.5
* pandas       V:0.23.4
* matplotlib   V:2.2.3
* scikit-learn V:0.19.2
<p>

##Examples
Examples of how to run the code will be added at a later date.
