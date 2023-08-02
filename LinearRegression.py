# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:20:15 2022

@author: Luís Almas 61726
@author: Duarte Belo 55793
"""

#%matplotlib inline
import pandas as pd
from pylab import *
import numpy as np
import random 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

FILE = "SatelliteConjunctionDataRegression.csv"

def load_data_from_file(fileName):
    return pd.read_csv(fileName)

def get_splited_Data(data):
    #shuffle data
    data.reindex(np.random.permutation(data.index))
    #split data
    return train_test_split(
        data, train_size=0.8,
        test_size=0.2
        )

def test_models(train_data, validation_data):
    x, y = train_data[:,:-1], train_data[:,-1]
    val_x, val_y = validation_data[:,:-1], validation_data[:,-1]

    errors_train = []
    errors_validation = []
    
    best_error = 10000000
    best_model = None
    best_degree = 1
    
    for degree in range(1,7):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(x)
        model = LinearRegression().fit(poly_features, y)
        errors_train.append(calc_error(model, poly_features, y, "blue"))
        error_validation = calc_error(model, poly.fit_transform(val_x), val_y, "red")
        
        if (error_validation < best_error):
            best_error = error_validation
            best_model = model
            best_degree = degree
        
        errors_validation.append(error_validation)
        plot([min(y),0,max(y)], [min(y),0,max(y)], color="grey")
        savefig('REGRESS-PRED-VS-TRUE-'+str(degree)+'.png')
        show()

    return errors_train, errors_validation, best_model, best_error, best_degree

def calc_error(model, x, y, color=""):
    predictions = model.predict(x)
    if(color != ""):
        plot(y, predictions, "or", marker=".", color=color)
    return mean_squared_error(y, predictions)
    

def run_3():
    degrees = [1,2,3,4,5,6]
    train_data, test_data = get_splited_Data(load_data_from_file(FILE))
    train_data, validation_data = train_data[:len(train_data)//2 + 1], train_data[len(train_data)//2:]
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    validation_data = scaler.transform(validation_data)
    
    train_errors, validation_errors, best_model, best_error, best_degree = test_models(train_data, validation_data)
    
    plt.figure()
    yscale("log")
    plt.ylabel("mse")
    plt.xlabel("degree")
    plot(degrees, train_errors, '-s', label="train", color="blue")  
    plot(degrees, validation_errors, '-x', label="val", color="red")
    plt.legend()
    savefig('REGRESS-TR-VAL.png')
    show()
    poly = PolynomialFeatures(degree=best_degree, include_bias=False)
    test_error = calc_error(best_model, poly.fit_transform(test_data[:,:-1]), test_data[:,-1])
    print('\nAccording to the test set the model can perform predictions with an error of +- '+ str(np.sqrt(test_error)) +' meters.')


run_3()





