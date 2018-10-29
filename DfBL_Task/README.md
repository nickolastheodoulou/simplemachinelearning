# Department for Big Lorries (DfBL) Data Analysis Task #

## Outline of Project ##

The DfBL conducts inspections on a sample of road-going vehicles. It aims to inspect each road vehicle once ever 12 and
most often inspects vehicles once every 6 years and tends to inspect vehicles found to be in poor condition more often.
It categorises the vehicles into three types; heavy goods vehicles, light goods vehicles and personnel; the 10 most 
common manufacturers and finally grades the vehicles on a 100-point scale where 100 is perfect and 0 represents a 
vehicle in hazardous condition. 

This is a repository that analysis the DfBL data set contains the attributes: VehicleID, FinancialYear, VehicleType, 
Manufacturer and ConditionScore for 30305 Vehicles.

## Description of the file structure ##

* The Report folder contains DfBL_Quantitative_Task_Report.pdf which is the brief of the task.
* main.py is where the main program is executed and calls methods from the DataExploration class.
* Class_Data_Loader.py contains the class DataLoader that loads in the data from a CSV file and stores it as an object
* Class_Data_Exploration.py contains the class DataExploration that inherits DataLoader. It's purpose is to print and 
plot various properties of the data so that it can be analysed
* Data_In contains VehicleData_csv.csv which is the data set that is analysed
* Data_Out contains all the plots that the code produces saved as a pdf file

## Packages Required: ##
Environment set up in python 3.7 and runs with following packages:
* numpy        
* pandas       
* matplotlib   
* seaborn

## How to run the code ##
Ensure all the packages are installed, set the working directory to DfBL_Task and run main.py. If there are any problems
please email me at: <nickolastheodoulou@hotmail.com>.