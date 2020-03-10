# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:58:23 2020

@author: Santosh Sah
"""

from SimpleLinearRegressionUtils import (readSimpleLinearRegressionModel, readSimpleLinearRegressionXTest, readSimpleLinearRegressionYTest,
                                         readSimpleLinearRegressionXTrain, readSimpleLinearRegressionYTrain)
"""
calculating metrics such as r-square, coefficient and y-intercept for training data

"""
def calculateMetricsForTrainingDataSet():
    
    X_train = readSimpleLinearRegressionXTrain()
    y_train = readSimpleLinearRegressionYTrain()
    simpleLinearRegressionModel = readSimpleLinearRegressionModel()
    
    #calculate r-square
    simpleLinearRegressionRSquare = simpleLinearRegressionModel.score(X_train, y_train)
    print(simpleLinearRegressionRSquare) #value of r-square .9411
    
    #calculate coefficient
    simpleLinearRegressionCoefficient = simpleLinearRegressionModel.coef_
    print(simpleLinearRegressionCoefficient) #value of coefficient is 9312.57
    
    #calculate y-intercept
    simpleLinearRegressionIntercept = simpleLinearRegressionModel.intercept_
    print(simpleLinearRegressionIntercept) #value of y-intercept is 26780.099

    
"""
calculating metrics such as r-square, coefficient and y-intercept for testing data

"""
def calculateMetricsForTestingDataSet():
    
    X_test = readSimpleLinearRegressionXTest()
    y_test = readSimpleLinearRegressionYTest()
    simpleLinearRegressionModel = readSimpleLinearRegressionModel()
    
    #calculate r-square
    simpleLinearRegressionRSquare = simpleLinearRegressionModel.score(X_test, y_test)
    print(simpleLinearRegressionRSquare) #value of r-square .9881
    
    #calculate coefficient
    simpleLinearRegressionCoefficient = simpleLinearRegressionModel.coef_
    print(simpleLinearRegressionCoefficient) #value of coefficient is 9314.57
    
    #calculate y-intercept
    simpleLinearRegressionIntercept = simpleLinearRegressionModel.intercept_
    print(simpleLinearRegressionIntercept) #value of y-intercept is 26780.099
    
if __name__ == "__main__":
    calculateMetricsForTrainingDataSet()
    #calculateMetricsForTestingDataSet()