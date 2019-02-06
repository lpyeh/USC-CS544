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
from tfidf import *
from preprocess import *

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

def get_vocab(text):
    vocab = []
    for doc in text:
        for word in tokenize(doc):
            if word not in stopwords:
                vocab.append(word)
    return list(set(vocab))


def fit(X, y_pos, y_truth):
    label_count = defaultdict(float)
    log_priors = defaultdict(float)
    word_counts = defaultdict(dict)
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
    
    # TODO: calculate tfidf vec for each document, then add THAT value instead of just count (rn it's a BoW rep.)
    for text, pos_label, truth_label in zip(X, y_pos, y_truth):
        label1 = 'pos' if pos_label == 1 else 'neg'
        label2 = 'truth' if truth_label == 1 else 'deceptive'
        doc_counts = dict(word_count(tokenize(text)))
        for word, count in doc_counts.items():
            if word in stopwords:
                continue;
            if word not in word_counts[label1]:
                word_counts[label1][word] = 0.0
            if word not in word_counts[label2]:
                word_counts[label2][word] = 0.0
            word_counts[label1][word] += count
            word_counts[label2][word] += count
    return word_counts, log_priors, label_count
    

def write(write_file):
    columns = ['word','positive', 'negative', 'truthful', 'deceptive']
    write_file.write('-------log priors--------\n')
    write_file.write('positive, negative, truthful, deceptive\n')
    log_list = [str(log_priors['pos']), str(log_priors['neg']), str(log_priors['truth']), str(log_priors['deceptive'])] 
    write_file.write(','.join(log_list))
    write_file.write('\n')
    write_file.write('----loglikelihoods for each word and label----\n')
    write_file.write(','.join(columns))
    write_file.write('\n')



def train():
    write_file = open('nbmodel.txt', 'w')
    write(write_file)

    loglikelihood = defaultdict(dict)
    for word in vocab:
        loglikelihood['pos'][word] = math.log((word_counts['pos'].get(word, 0.0) + 1) / (label_count['pos'] + len(vocab)))
        loglikelihood['neg'][word] = math.log((word_counts['neg'].get(word, 0.0) + 1) / (label_count['pos'] + len(vocab)))
        loglikelihood['truth'][word] = math.log((word_counts['truth'].get(word, 0.0) + 1) / (label_count['pos'] + len(vocab)))
        loglikelihood['deceptive'][word] = math.log((word_counts['deceptive'].get(word, 0.0) + 1) / (label_count['pos'] + len(vocab)))
        row = [word, str(loglikelihood['pos'][word]), str(loglikelihood['neg'][word]), str(loglikelihood['truth'][word]), str(loglikelihood['deceptive'][word])]
        write_file.write(','.join(row))
        write_file.write('\n')
        write_file.flush()
    write_file.close()


if __name__=='__main__':
    input_path = sys.argv[-1]
    
    X_train, y_train_pos, y_train_truth = load_data()

    stopwords = get_stopwords(X_train)
    vocab = get_vocab(X_train)
    word_counts, log_priors, label_count = fit(X_train, y_train_pos, y_train_truth)
    train()
    


