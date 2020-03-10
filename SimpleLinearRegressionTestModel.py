# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:41:21 2020

@author: Santosh Sah
"""

from SimpleLinearRegressionUtils import (readSimpleLinearRegressionXTest, readSimpleLinearRegressionModel)

"""
test the model on testing dataset
"""
def testSimpleLinearRegressionModel():
    
    X_test = readSimpleLinearRegressionXTest()
    simpleLinearRegressionModel = readSimpleLinearRegressionModel()
    
    y_pred = simpleLinearRegressionModel.predict(X_test)
    print(y_pred)

if __name__ == "__main__":
    testSimpleLinearRegressionModel()
