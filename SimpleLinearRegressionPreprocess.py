# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:01:53 2020

@author: Santosh Sah
"""

from SimpleLinearRegressionUtils import importLinearRegressionDataset, saveTrainingAndTestingDataset

def preprocess():
    
    X_train, X_test, y_train, y_test = importLinearRegressionDataset("Simple_Linear_Regression_Salary_Data.csv")
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()