# Predicting Sales of a customer #

## Outline of Project ##
The objective of this repository is to predict whether there will be a sale (Yes/No) of car insurance using the data set.
There are 10 attributes in the dataset including: Veh_Mileage, Credit_Score, Licesnse_Length, Veh_Value, Price, Age,
Marital_Status, Tax, Date. There are 50,000 entries.

## Description of the file structure ##

* The code is all contained in the main directory
* main.py is where the main program is executed and calls methods from the DataModeler class.
* Class_Data_Loader.py contains the class DataLoader that loads in the data from a CSV file and stores it as an object
* Class_Data_Exploration.py contains the class DataExploror that inherits DataLoader. It's purpose is to print and 
plot various properties of the data so that it can be analysed
* Class_Data_Preprocessor contains the class DataPreprocessor that preprocesses the data
using various methods such as normalising and one-hot encoding
* Class_Data_Modeler contains Data_Modeler which models the data and performs cross
validation
* Data_In contains DS_Assessment.csv which is the data set that is analysed
* Data_Out contains all various plots that the code produces saved as a pdf file

This was done by using inheritance where DataModeler -> DataPreprocessor -> DataExploror -> DataLoader and the arrow 
 is a symbol indicating 'inherited from'. 

## Packages Required: ##
Environment set up in python 3.5 and runs with following packages:
* numpy        
* pandas       
* matplotlib   
* seaborn
* sklearn
* scipy

## Example of the code ##
main.py demonstrates how the various methods can be called to be used correctly. It also demonstrates some of my thought 
processes when analysing and pre-processing the data with comments. 

## How to run the code ##
Ensure all the packages are installed, set the working directory to Car_Insurance_Sales_Prediction and run main.py. If 
there are any problems please email me at: <nickolastheodoulou@hotmail.com>.

