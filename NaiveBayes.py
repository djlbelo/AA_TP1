# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:30:51 2022

@author: Luís Almas 61726
@author: Duarte Belo 55793
"""

#%matplotlib inline
from pylab import *
import numpy as np
import random 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

TRAIN_FILE = 'TP1_train.tsv'
TEST_FILE = 'TP1_test.tsv'

def get_data(file):
    data = np.genfromtxt(fname=file, delimiter="\t")
    np.random.shuffle(data)
    return data

def gaussian_bayes(train_set, test_set):
    train_y = train_set[:,-1]
    train_x = train_set[:,:-1]
    test_x = test_set[:,:-1]
    test_y = test_set[:,-1]
    gauss = GaussianNB()
    gauss.fit(train_x, train_y)
    gb_prediction = gauss.predict(test_x)
    gb_error = 1 - gauss.score(test_x, test_y)
    
    return gb_prediction, gb_error
    
def set_naive_bayes(train_set, test_set, train_set_aux):
    print('\nCalculating the best bandwith...')
    n_folds = 5
    fold = StratifiedKFold(n_splits=n_folds)
    scaler = StandardScaler()
    best_bandwith = 0
    best_error = 1000
    bandwiths = []
    train_errors = []
    validation_errors = []
    
    for bandwidth in np.arange(0.02, 0.6, 0.02):        
        train_error_total = 0
        validation_error_total = 0
        bandwiths.append(bandwidth)
        
        for train_index, validation_index in fold.split(train_set[:,:-1], train_set[:,-1]):
            train_error, validation_error = calculate_fold(train_set, train_set_aux, train_index, validation_index, bandwidth)
            train_error_total += train_error
            validation_error_total += validation_error
            
        train_errors.append(train_error_total/n_folds)
        validation_errors.append(validation_error_total/n_folds)
            
        if (validation_error_total/n_folds < best_error):
            best_error = validation_error_total/n_folds
            best_bandwith = bandwidth
    
    
    nb_prediction, nb_test_error = naive_bayes_test_error(train_set, test_set, best_bandwith)

    return nb_prediction, best_bandwith, nb_test_error, bandwiths, train_errors, validation_errors
    
def naive_bayes_test_error(train_set, test_set, bandwidth):
    train_y = train_set[:,-1]
    train_x = train_set[:,:-1]
    classDistribuiton = np.unique(train_y)
    result = []
    
    for data_class in classDistribuiton:
        x_from_class = train_x[train_y == data_class]
        
        #Probability of class
        prob = np.log(len(x_from_class)/len(train_x))
        
        #For each atribute
        for atribute_index in range(x_from_class.shape[1]):
            prob += score_attribute(atribute_index, x_from_class, test_set[:,:-1], bandwidth)
            
        result.append(prob)
        
    #get max index from each row to define if 1 or 0
    prediction = np.argmax(result, axis=0)
    return prediction, 1 - accuracy_score(test_set[:,-1], prediction, normalize=True)
        
def calculate_fold(train_set, train_set_aux, train_index, validation_index, bandwidth):
    #classes -> 1, -1
    train_y = train_set[:,-1]
    train_x = train_set[:,:-1]
    classDistribuiton = np.unique(train_y)
    result = []
    
    
    for data_class in classDistribuiton:
        x_from_class = train_x[train_index][train_y[train_index] == data_class]
        
        #Probability of class
        prob = np.log(len(x_from_class)/len(train_x[train_index]))
        
        #For each atribute
        for atribute_index in range(x_from_class.shape[1]):
            prob += score_attribute(atribute_index, x_from_class, train_x, bandwidth)
            
        result.append(prob)
            
       
    #get max index from each row to define if 1 or 0
    prediction = np.argmax(result, axis=0)
    return calculate_errors(prediction, train_index, validation_index, train_y, train_set_aux[:,-1])
            
        
def calculate_errors(prediction, train_index, validation_index, y, y_aux):
    return 1 - accuracy_score(y[train_index], prediction[train_index], normalize=True), 1 - accuracy_score(y_aux[validation_index], prediction[validation_index], normalize=True)
        
def score_attribute(atribute_index, x_from_class, train_x, bandwidth):
    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kernel.fit(x_from_class[:, [atribute_index]])
    return kernel.score_samples(train_x[:, [atribute_index]])

def mc_nemar_test(result1, result2, test_y):

    yes_no = 0;
    no_yes = 0;

    for y_index in range(0, len(test_y)-1):
        if (result1[y_index] == test_y[y_index] and result2[y_index] != test_y[y_index]):
            yes_no += 1
        if (result1[y_index] != test_y[y_index] and result2[y_index] == test_y[y_index]):
            no_yes += 1

    return abs(no_yes - yes_no - 1)**2 / (no_yes + yes_no)
  
def normal_test(prediction, test_y):
    x = sum(prediction != test_y)
    n = len(test_y)
    N_p0 = n*(x/n)
    sigma = math.sqrt(N_p0*(1-(x/n)))
    value_interval = sigma*1.96
    min_int = N_p0 - value_interval
    max_int = N_p0 + value_interval

    return min_int, N_p0, max_int
  
def printer(train_y, test_y, gb_prediction, nb_prediction, train_errors, validation_errors, nb_test_error, gb_error, best_bandwith, bandwiths):
    print("====================== ERROR ======================")
    print("Naive Bayes = ", (1.0-metrics.accuracy_score(test_y[:-1],nb_prediction))*100, "%")
    print("Gaussian Naive Bayes = ", (1.0-metrics.accuracy_score(test_y[:-1],gb_prediction))*100, "%\n")
    
    print("====================== ACCURACY ======================")
    print("Naive Bayes = ", (metrics.accuracy_score(test_y[:-1],nb_prediction))*100, "%")
    print("Gaussian Naive Bayes = ", (metrics.accuracy_score(test_y[:-1],gb_prediction))*100, "%\n")
    
    print("====================== BEST BANDWIDTH ======================")
    print('Best bandwith = ', best_bandwith, "\n")
    
    nbg_low, nbg_0, nbg_high = normal_test(gb_prediction, test_y)
    nb_low, nb_0, nb_high = normal_test(nb_prediction, test_y)
    
    print("====================== Normal Test ======================")
    print("Naive Bayes = ", nb_low, " < ", nb_0, " < ", nb_high)
    print("Gaussian Naive Bayes = ", nbg_low, " < ", nbg_0, " < ", nbg_high, "\n")
    
    print("====================== McNeman's Test ======================")
    print("Gaussian Naive Bayes VS Naive Bayes = ", mc_nemar_test(gb_prediction, nb_prediction, test_y), "\n")
    
    plt.figure()
    plt.ylabel("Error")
    plt.xlabel("Bandwith")
    plot(bandwiths, train_errors, label="Training Error")
    plot(bandwiths, validation_errors, label="Validation Error")
    plt.legend()
    savefig('NB.png')
    show()
    
    
def run():
    # x -> data[:,:-1], y -> data[:,-1]
    train_set = get_data(TRAIN_FILE)
    test_set = get_data(TEST_FILE)
    train_set_aux = train_set.copy()
    scaler = StandardScaler()
    train_set[:,:-1] = scaler.fit_transform(train_set[:,:-1])
    train_set_aux[:,:-1] = scaler.transform(train_set_aux[:,:-1])
    test_set[:,:-1] = scaler.transform(test_set[:,:-1])


    train_y = train_set[:,-1]
    test_y = test_set[:,-1]

    nb_prediction, best_bandwith, nb_test_error, bandwiths, train_errors, validation_errors = set_naive_bayes(train_set, test_set[:-1], train_set_aux)
    gb_prediction, gb_error = gaussian_bayes(train_set, test_set[:-1])

    printer(train_y, test_y, gb_prediction, nb_prediction, train_errors, validation_errors, nb_test_error, gb_error, best_bandwith, bandwiths)
