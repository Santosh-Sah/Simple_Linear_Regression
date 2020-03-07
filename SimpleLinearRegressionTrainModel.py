# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:25:38 2020

@author: Santosh Sah
"""
from sklearn.linear_model import LinearRegression
from SimpleLinearRegressionUtils import (saveSimpleLinearRegressionModel, readSimpleLinearRegressionXTrain, readSimpleLinearRegressionYTrain)

"""
Train simple linear regression model 
"""
def trainSimpleLinearRegressionModel():
    
    X_train = readSimpleLinearRegressionXTrain()
    y_train = readSimpleLinearRegressionYTrain()
    
    simpleLinearRegression = LinearRegression()
    simpleLinearRegression.fit(X_train, y_train)
    
    saveSimpleLinearRegressionModel(simpleLinearRegression)

if __name__ == "__main__":
    trainSimpleLinearRegressionModel()    
