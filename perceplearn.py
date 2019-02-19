#!usr/bin/python
import sys
import math
from collections import *
import glob
from preprocess import *
import os
import json
import numpy as np
import sklearn
from sklearn.metrics import precision_score as precision, recall_score as recall, f1_score as f1
"""
Training file for Perceptron program for CSCI 544.

Author: Leigh Yeh
Date: 1/31/2019
University of Southern California
"""


def load_data(path):
    print("Loading Data")
    all_files = glob.glob(os.path.join(path, '*/*/*/*.txt'))
    train = []
    positive_train = []
    truthful_train = []

    for file_ in all_files:
        class1, class2, fold, fname = file_.split('/')[-4:]
        text = open(file_).read()
        train.append(tokenize(clean(text)))
        
        positive_train.append(1) if 'positive' in class1 else positive_train.append(-1)
        truthful_train.append(1) if 'truthful' in class2 else truthful_train.append(-1)

    positive_train = np.array(positive_train, dtype = np.int32)
    # train = np.array(train, np.float32)
    # return train, positive_train, truthful_train
    return train, positive_train


def vectorize(X, vocab, method):
    print("Vectorizing Data")
    if method == 'count':
        data = word_count(X, vocab)
    elif method == 'tfidf':
        data = np.array(tfidf(X, vocab), dtype=np.float32)
    return data

"""
def shuffle(X):
"""


def vanilla_train(X, y_train, num_epochs=40):
    print("-----Vanilla Perceptron----")
    epoch = 0
    # TODO: length is definitely ont len(X[0]), need to figure this out
    num_docs = len(X)
    w = np.zeros(len(vocab))
    bias = 0

    for epoch in range(num_epochs):
        print("epoch: {}".format(epoch))

        for x, y in zip(X, y_train):
            
            activation = np.sum(np.multiply(x, w)) + bias
            
            if y * activation > 0:
                continue
            else:
                w += np.multiply(x, y)
                # TODO: Do we need to loop through this whole thing again??
                bias += y
    return w, bias


def averaged_train(X, y_train, num_epochs=40):
    print("-------Averaged Perceptron--------")
    weights = np.zeros(len(vocab))
    u = np.zeros(len(vocab))
    count = 1
    bias = 0
    beta = 0

    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        for x, y in zip(X, y_train):
            if y * (np.sum(np.multiply(weights, x)) + bias) <= 0:
                weights += np.multiply(x, y)
                u += np.multiply(x, y) * count
                bias += y
                beta += (y * count)
            count += 1
    avg_w = weights - (u/count)
    avg_b = bias - (beta/count)
    return avg_w, avg_b


def predict(X_test, weights, bias):
    result = []
    for x in X_test:
        activation = np.sum(np.multiply(x, weights)) + bias
        if activation > 0:
            result.append(1)
        else:
            result.append(-1)
    return result



    write_file = open(write_file, 'w')
    json.dump([log_priors, label_count, vocab, word_counts], write_file, indent=2)
    write_file.close()

if __name__=='__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    X_train, y_train_pos = load_data(input_path)
    X_test, y_test_pos = load_data(output_path)
    vocab = get_vocab(X_train, vocab_length=2000)
    data = vectorize(X_train, vocab, method='count')
    test = vectorize(X_test, vocab, method='count')

    #------vanilla-------#
    weights_vanilla, bias_vanilla = vanilla_train(data, y_train_pos) 
    vanilla_results = predict(test, weights_vanilla, bias_vanilla)
    

    #------averages-------#
    averaged_weights, averaged_bias = averaged_train(data, y_train_pos)
    avg_results = predict(test, averaged_weights, averaged_bias)

    print("-----Vanilla Results------")
    print("recall: {} | precision: {} | f1: {}".format(recall(y_test_pos, vanilla_results), precision(y_test_pos, vanilla_results), f1(y_test_pos, vanilla_results)))
    print("-----Avg Results-------")
    print("recall: {} | precision: {} | f1: {}".format(recall(y_test_pos, avg_results), precision(y_test_pos, avg_results), f1(y_test_pos, avg_results)))



