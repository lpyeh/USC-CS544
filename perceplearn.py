#!usr/bin/python
import sys
import math
from collections import *
import glob
from preprocess import *
import os
import json
import numpy

"""
Training file for Naive Bayes program for CSCI 544.

Author: Leigh Yeh
Date: 1/31/2019
University of Southern California
"""

def load_data():
    all_files = glob.glob(os.path.join(input_path, '*/*/*/*.txt'))
    train = []
    positive_train = []
    truthful_train = []

    for file_ in all_files:
        class1, class2, fold, fname = file_.split('/')[-4:]
        text = open(file_).read()
        train.append(text)
        positive_train.append(1) if 'positive' in class1 else positive_train.append(-1)
        truthful_train.append(1) if 'truthful' in class2 else truthful_train.append(-1)

    train = clean(train)
    return train, positive_train, truthful_train

def shuffle(data):



def vanilla_train(X, y, learning_rate=0.1, num_epochs=100):
    epoch = 0
    # TODO: length is definitely ont len(X[0]), need to figure this out
    w = np.zeros(len(X[0]))
    num_docs = len(X)
    bias = 0

    for epoch in range(num_epochs):
        for i in range(num_docs):
            x = X[i]
            y = y[i]
            
            activation = np.dot(w[i], x) + b
            
            if y * activation > 0:
                continue
            else:
                # TODO: Do we need to loop through this whole thing again??
                for j in range(num_docs):
                    w[j] = w[j] + (y * x[j])
                bias = bias + y
    return w, bias


def average_train(X, y, learning_rate = 0.1, num_epochs=100):
    weights = np.zeros(1)
    beta = np.zeros(1)
    weight_mat = np.zeros(2)
    idx = 0
    count = 1
    epoch = 1
    bias = 0

    while epoch < range(num_epochs):
        for i in range(len(X)):
            x = X[i]
            y = y[i]
            if y * (np.dot(weights, x)) > 0:
                count += 1
            else:
                idx += 1
                weights_mat[idx,] = 
                weights[ = w + (y * x)
                bias = bias + y
                cached = cached + (y * count * x)
                beta = beta + (y * count)
                epoch += 1
            # count += 1

    return 
                # ENDED HERE

def test(weights, bias, X_test):
    result = []
    for x, w in zip(X_test, weights):
        if np.dot(w, x) < 0:
            result.append(0)
        else:
            result.append(1)
    return result



def write(write_file):
    write_file = open(write_file, 'w')
    json.dump([log_priors, label_count, vocab, word_counts], write_file, indent=2)
    write_file.close()

if __name__=='__main__':
    input_path = sys.argv[-1]
    X_train, y_train_pos, y_train_truth = load_data()
    word_counts, log_priors, label_count, vocab = train(X_train, y_train_pos, y_train_truth)
    write('nbmodel.txt')


