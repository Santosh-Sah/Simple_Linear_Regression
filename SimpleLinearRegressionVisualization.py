# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:48:17 2020

@author: Santosh Sah
"""
import matplotlib.pyplot as plt
from SimpleLinearRegressionUtils import (readSimpleLinearRegressionModel, readSimpleLinearRegressionXTrain, readSimpleLinearRegressionYTrain,
                                         readSimpleLinearRegressionXTest, readSimpleLinearRegressionYTest)
"""
Visualizing training set results
"""
def visualisingTrainingSetResult():
    
    X_train = readSimpleLinearRegressionXTrain()
    y_train = readSimpleLinearRegressionYTrain()
    simpleLinearRegressionModel = readSimpleLinearRegressionModel()
    
    plt.scatter(X_train, y_train, color = "red")
    plt.plot(X_train, simpleLinearRegressionModel.predict(X_train), color = "blue")
    plt.title("Salary Vs Experiences (Training Set)")
    plt.xlabel("Year of Experience")
    plt.ylabel("Salary")
    
    plt.savefig("trainingsetresult.png")
    
    plt.show()

"""
Visualizing testing set results
"""
def visualisingTestingSetResult():
    
    X_test = readSimpleLinearRegressionXTest()
    y_test = readSimpleLinearRegressionYTest()
    X_train = readSimpleLinearRegressionXTrain()
    simpleLinearRegressionModel = readSimpleLinearRegressionModel()
    
    plt.scatter(X_test, y_test, color = "red")
    plt.plot(X_train, simpleLinearRegressionModel.predict(X_train), color = "blue")
    plt.title("Salary Vs Experiences (Test Set)")
    plt.xlabel("Year of Experience")
    plt.ylabel("Salary")
    
    plt.savefig("testingsetresult.png")
    
    plt.show()

if __name__ == "__main__":
    #visualisingTrainingSetResult()
    visualisingTestingSetResult()
    
    