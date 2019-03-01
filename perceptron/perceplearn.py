#!usr/bin/python
import sys
import math
from collections import *
import glob
from preprocess import *
import os
import json
import numpy as np
import random

"""
Training file for Perceptron program for CSCI 544.

Author: Leigh Yeh
Date: 2/19/2019
University of Southern California
"""


def load_data(path):
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

    positive_train = np.array(positive_train, dtype = np.float32)
    truthful_train = np.array(truthful_train, dtype = np.float32)

    # shuffle
    return train, truthful_train, positive_train


def vectorize(X, vocab, method):
    # print("Vectorizing Data")
    if method == 'count':
        data = word_count(X, vocab)
    elif method == 'tfidf':
        data = np.array(tfidf(X, vocab), dtype=np.float32)
    return data


def shuffle(X, y1, y2):
    data = list(zip(X, y1, y2))
    random.shuffle(data)
    train, truthful_train, positive_train = zip(*data)
    return train, truthful_train, positive_train


def vanilla_train(X, y_train, num_epochs=40):
    epoch = 0
    num_docs = len(X)
    w = np.zeros(len(vocab))
    bias = 0.0

    for epoch in range(num_epochs):
        for x, y in zip(X, y_train):
            
            if y * (np.sum(np.multiply(w, x)) + bias) > 0:
                continue
            else:
                w += np.multiply(x, y)
                bias += (y * 1.0)
    return w, bias


def averaged_train(X, y_train, num_epochs=40):
    weights = np.zeros(len(vocab))
    u = np.zeros(len(vocab))
    count = 1.0
    bias = 0.0
    beta = 0.0

    for epoch in range(num_epochs):
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


if __name__=='__main__':
    input_path = sys.argv[-1]
    X_train, y_train_truth, y_train_pos = load_data(input_path)
    vocab = get_vocab(X_train, vocab_length=2000)
    
    data = vectorize(X_train, vocab, method='count')
    data, y_train_truth, y_train_pos = shuffle(data, y_train_truth, y_train_pos)
    
    w_vanilla_pos, bias_vanilla_pos = vanilla_train(data, y_train_pos)
    w_vanilla_truth, bias_vanilla_truth = vanilla_train(data, y_train_truth)
    
    #------vanilla-------#
    vanilla_dict = defaultdict()
    vanilla_dict['positive'] = w_vanilla_pos.tolist()
    vanilla_dict['truthful'] = w_vanilla_truth.tolist()

    vanilla_dict['pos_bias'] = float(bias_vanilla_pos)
    vanilla_dict['truth_bias'] = float(bias_vanilla_truth)

    vanilla_dict['vocab'] = vocab
    
    out = open('vanillamodel.txt', 'w')
    json.dump(vanilla_dict, out, indent=2)
    out.close()
    
    
    #------averaged-------#
    avg_dict = defaultdict()
    w_avg_pos, bias_avg_pos = averaged_train(data, y_train_pos)
    w_avg_truth, bias_avg_truth = averaged_train(data, y_train_truth)
    
    avg_dict['positive'] = w_avg_pos.tolist()
    avg_dict['truthful'] = w_avg_truth.tolist()

    avg_dict['pos_bias'] = float(bias_avg_pos)
    avg_dict['truth_bias'] = float(bias_avg_truth)

    avg_dict['vocab'] = vocab

    out = open('averagedmodel.txt', 'w')
    json.dump(avg_dict, out, indent=2)
    out.close()

