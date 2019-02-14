#!usr/bin/python
import sys
import math
from collections import *
import glob
from preprocess import *
import os
import json

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
        positive_train.append(1) if 'positive' in class1 else positive_train.append(0)
        truthful_train.append(1) if 'truthful' in class2 else truthful_train.append(0)

    train = clean(train)
    return train, positive_train, truthful_train

def train(X, y_pos, y_truth):
    label_count = defaultdict(float)
    log_priors = defaultdict(float)
    word_counts = defaultdict(dict)
    vocab = defaultdict(float)
    n = len(X)

    count_pos = Counter(y_pos)
    count_truth = Counter(y_truth)
    label_count['pos'] = count_pos[1]
    label_count['neg'] = count_pos[0]
    label_count['truth'] = count_truth[1]
    label_count['deceptive'] = count_truth[0]

    log_priors['pos'] = math.log(label_count['pos'] / n)
    log_priors['neg'] = math.log(label_count['neg'] / n)
    log_priors['truth'] = math.log(label_count['truth'] / n)
    log_priors['deceptive'] = math.log(label_count['deceptive'] / n)
    
    for text, pos_label, truth_label in zip(X, y_pos, y_truth):
        label1 = 'pos' if pos_label == 1 else 'neg'
        label2 = 'truth' if truth_label == 1 else 'deceptive'
        doc_counts = dict(word_count(tokenize(text)))
        for word, count in doc_counts.items():
            if word not in vocab:
                vocab[word] = 0.0
            if word not in word_counts[label1]:
                word_counts[label1][word] = 0.0
            if word not in word_counts[label2]:
                word_counts[label2][word] = 0.0
            word_counts[label1][word] += count
            word_counts[label2][word] += count
            vocab[word] += count
    return word_counts, log_priors, label_count, vocab
    

def write(write_file):
    write_file = open(write_file, 'w')
    json.dump([log_priors, label_count, vocab, word_counts], write_file, indent=2)
    write_file.close()

if __name__=='__main__':
    input_path = sys.argv[-1]
    X_train, y_train_pos, y_train_truth = load_data()
    word_counts, log_priors, label_count, vocab = train(X_train, y_train_pos, y_train_truth)
    write('nbmodel.txt')


