#!usr/bin/python
import sys
import math
import os
import numpy as np
from collections import defaultdict
from collections import Counter
import glob
import re
import string

"""
Training file for Naive Bayes program for CSCI 544.

Author: Leigh Yeh
Date: 1/31/2019
University of Southern California
"""

def load_data():
    all_files = glob.glob(os.path.join(input_path, '*/*/*/*.txt'))
    train = []
    test = []
    
    # text = []
    positive_test = []
    truthful_test = []
    positive_train = []
    truthful_train = []

    # print(all_files)

    for file_ in all_files:
        # print(file_)
        class1, class2, fold, fname = file_.split('/')[-4:]
        text = open(file_).read()
        # remove the if statement to submit to Vocareum
        if fold == 'fold1':
            test.append(text)
            # test.append(text)
            positive_test.append(1) if 'positive' in file_ else positive_test.append(0)
            truthful_test.append(1) if 'truthful' in file_ else truthful_test.append(0)
        else:
            train.append(text)
            positive_train.append(1) if 'positive' in file_ else positive_train.append(0)
            truthful_train.append(1) if 'truthful' in file_ else truthful_train.append(0)

    return train, positive_train, truthful_train, test, positive_test, truthful_test


def tokenize(text):
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator).lower()
    return re.split("\W+", text)


def word_count(text):
    text = list(text)
    return Counter(text)


def train(X, y_pos, y_truth):
    label_count = defaultdict(int)
    log_priors = defaultdict(int)
    # TODO: make this a (label, word, count) tuple mabye?
    word_counts = defaultdict(dict)
    vocab = set()
    n = len(X)
    
    count_pos = Counter(y_pos)
    count_truth = Counter(y_truth)
    # TODO: Add other label as well? Just do it all in one pass?
    label_count['pos'] = count_pos[1]
    label_count['neg'] = count_pos[0]
    label_count['truth'] = count_truth[1]
    label_count['deceptive'] = count_truth[0]

    log_priors['pos'] = math.log(label_count['pos'] / n)
    log_priors['neg'] = math.log(label_count['neg'] / n)
    log_priors['truth'] = math.log(label_count['truth'] / n)
    log_priors['deceptive'] = math.log(label_count['deceptive'] / n)

    for text, labels in zip(X, y_pos):
        label = 'pos' if labels == 1 else 'neg'
        counts = word_count(tokenize(text))

        for word, count in counts.items():
            vocab.add(text)
            if word not in word_counts[label]:
                word_counts[label][word] = 0
            else:
                word_counts[label][word] += count

    for text, labels in zip(X, y_truth):
        label = 'truth' if labels == 1 else 'deceptive'
        counts = word_count(tokenize(text))

        for word, count in counts.items():
            vocab.add(text)
            if word not in word_counts[label]:
                word_counts[label][word] = 0
            else:
                word_counts[label][word] += count

def predict(X):
    result = []
    for x in X:
        counts = word_counts(tokenize(x))



if __name__=='__main__':

    input_path = sys.argv[1]
    X_train, y_train_pos, y_train_truth, X_test, y_test_pos, y_test_truth = load_data()
    train(X_train, y_train_pos, y_train_truth)
