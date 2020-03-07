# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:12:21 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importLinearRegressionDataset(linearRegressionDatasetFileName):
    
    linearRegressionDataset = pd.read_csv(linearRegressionDatasetFileName)
    X = linearRegressionDataset.iloc[:, :-1].values
    y = linearRegressionDataset.iloc[:, 1].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveSimpleLinearRegressionStandardScaler():
    
    simpleLinearRegressionStandardScalar = StandardScaler()
    
    #Write SimpleLinearRegressionStandardScaler in a picke file
    with open("SimpleLinearRegressionStandardScaler.pkl",'wb') as SimpleLinearRegressionStandardScaler_Pickle:
        pickle.dump(simpleLinearRegressionStandardScalar, SimpleLinearRegressionStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save SimpleLinearRegressionModel as a pickle file.
"""
def saveSimpleLinearRegressionModel(simpleLinearRegressionModel):
    
    #Write SimpleLinearRegressionModel as a picke file
    with open("SimpleLinearRegressionModel.pkl",'wb') as SimpleLinearRegressionModel_Pickle:
        pickle.dump(simpleLinearRegressionModel, SimpleLinearRegressionModel_Pickle, protocol = 2)

"""
read SimpleLinearRegressionStandardScalar from pickel file
"""
def readSimpleLinearRegressionStandardScaler():
    
    #load SimpleLinearRegressionStandardScaler object
    with open("SimpleLinearRegressionStandardScaler.pkl","rb") as SimpleLinearRegressionStandardScaler:
        simpleLinearRegressionStandardScalar = pickle.load(SimpleLinearRegressionStandardScaler)
    
    return simpleLinearRegressionStandardScalar

"""
read simpleLinearRegressionModel from pickle file
"""
def readSimpleLinearRegressionModel():
    
    #load simpleLinearRegressionModel model
    with open("simpleLinearRegressionModel.pkl","rb") as SimpleLinearRegressionModel:
        simpleLinearRegressionModel = pickle.load(SimpleLinearRegressionModel)
    
    return simpleLinearRegressionModel

"""
read X_train from pickle file
"""
def readSimpleLinearRegressionXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readSimpleLinearRegressionXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readSimpleLinearRegressionYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readSimpleLinearRegressionYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test


