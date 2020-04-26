from datetime import date
from itertools import compress
from math import sqrt
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sklearn.mixture
from sklearn import metrics
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from metrics import compute_metrics


def cross_validation(regressor, data_samples, sample_classes, folds, histograms_boolean):

    results_for_each_round = []
    fited_model_for_each_round = []


    # SPLIT SAMPLES
    fold_index_array = split_samples((len(sample_classes), folds))
    
    
    for k in range(folds):
        
        train_index = np.arange(len(data_samples))[fold_index_array != k]
        test_index = np.arange(len(data_samples))[fold_index_array == k]

        train_samples = data_samples[train_index]
        test_samples = data_samples[test_index]
        
        train_labels = [sample_classes[i] for i in train_index]
        test_labels = [sample_classes[i] for i in test_index]
        print("Train set size: " + str(len(train_index)))
        print("Test  set size: " + str(len(test_index)))

        train_histograms = data_historgrams(np.asarray(train_labels))
        test_histograms = data_historgrams(np.asarray(test_labels))
        plot_data_histograms(
            train_histograms, test_histograms, k + 1, folds, histograms_boolean
        )
        
        
        # train model
        fited_model = fit_regressor(regressor,train_samples,train_labels)
        
        # test model
        round_results, result_names = test_regressor(regresor, test_samples,test_labels)
        
        
        # Store regressor and results
        fited_model_for_each_round.append(fited_model)#al final tindra tamany RxK amb cada element [count] tindrà un regressor per cada ronda
        results_for_each_round.append(round_results)
    
    
    return fited_model_for_each_round, results_for_each_round, result_names

def split_samples(number_of_samples, folds):

    fold_indexes = []
    samples_per_fold = int(number_of_samples) / folds)
    [[fold_indexes.append(n) for i in range(samples_per_fold)] for n in range(folds)]

    return np.asarray(fold_indexes)

def fit_regressor(regresor,train_samples,train_labels):
    if len(train_labels[0])>1:
        #if there is more than one label for each sample
        regressor = MultiOutputRegressor(regresor)
        regressor.fit(train_samples, (train_labels))
    else:
        # if there is just one label for each sample
        regressor.fit(train_samples, (train_labels))

    return regressor

def test_regressor(regresor, test_samples,test_labels):
    predicted_labels = regresor.predict(test_samples)
    compute_metrics(test_samples,test_labels)

    return round_results, result_names

