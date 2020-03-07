# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:20:25 2020

@author: Santosh Sah
"""
import pandas as pd
from SimpleLinearRegressionUtils import readSimpleLinearRegressionModel

def predict():
    
    simpleLinearRegression = readSimpleLinearRegressionModel()
    
    inputValue = [55]
    inputValueDataframe = pd.DataFrame(inputValue)
    
    predictedValue = simpleLinearRegression.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()