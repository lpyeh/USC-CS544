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
        # print(class1, class2)
        # remove the if statement to submit to Vocareum
        if fold == 'fold3':
            # print(file_)
            test.append(text)
            positive_test.append(1) if 'positive' in class1 else positive_test.append(0)
            truthful_test.append(1) if 'truthful' in class2 else truthful_test.append(0)
        else:
            train.append(text)
            positive_train.append(1) if 'positive' in class1 else positive_train.append(0)
            truthful_train.append(1) if 'truthful' in class2 else truthful_train.append(0)

    train = clean(train)
    test = clean(test)
    return train, positive_train, truthful_train, test, positive_test, truthful_test


def get_vocab(text):
    vocab = []
    for word in text:
        if word not in stopwords:
            vocab.append(word)
    return list(set(vocab))


def train(X, y_pos, y_truth):
    label_count = defaultdict(float)
    log_priors = defaultdict(float)
    # TODO: make this a (label, word, count) tuple mabye?
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
        # tfidf_vec = tfidf(tokenize(text), X, vocab)
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


def classify(X, word_counts, log_priors):
    # TODO: Need word_counts, log_priors to be output to txt file
    # outfile = open('output.txt', 'w')
    
    pos_score, neg_score, truth_score, dec_score = 0.0, 0.0, 0.0, 0.0
    truth_result = []
    pos_result = []
    for x in X:
        counts = word_count(tokenize(x))

        for word in counts.keys():
            if word in stopwords:
                continue;
            log_pos = math.log10((word_counts['pos'].get(word, 0.0) + 1) / (label_count['pos'] + len(vocab)))
            log_neg = math.log10((word_counts['neg'].get(word, 0.0) + 1) / (label_count['neg'] + len(vocab)))
            log_truth = math.log10((word_counts['truth'].get(word, 0.0) + 1) / (label_count['truth'] + len(vocab)))
            log_dec = math.log10((word_counts['deceptive'].get(word, 0.0) + 1) / (label_count['deceptive'] + len(vocab)))
            # print(word)

            pos_score += log_pos
            neg_score += log_neg
            truth_score += log_truth
            dec_score += log_dec

        pos_score += log_priors['pos']
        neg_score += log_priors['neg']
        truth_score += log_priors['truth']
        dec_score += log_priors['deceptive']

        if pos_score > neg_score:
           pos_result.append(1)
        elif neg_score > pos_score:
            pos_result.append(0)
        else:
            pos_result.append(1)

        if truth_score > dec_score:
            truth_result.append(1)
        elif dec_score > truth_score:
            truth_result.append(0)
        else:
            truth_result.append(1)
    # print(truth_result)
    return pos_result, truth_result


def score():
    tp1, fp1, tn1, fn1 = 0,0,0,0
    tp2, fp2, tn2, fn2 = 0, 0,0,0
    for i in range(len(result1)):
        if result1[i] == y_test_pos[i]:
            if result1[i] == 1:
                tp1 += 1
            else:
                tn1 += 1
        else:
            if result1[i] == 1:
                fp1 += 1
            else:
                fn1 += 1
    for i in range(len(result2)):
        if result2[i] == y_test_truth[i]:
            if result2[i] == 1:
                tp2 += 1
            else:
                tn1 += 1
        else:
            if result2[i] == 1:
                fp2 += 1
            else:
                fn2 += 1
    prec1 = tp1 / (tp1 + fp1)
    rec1 = tp1 / (tp1 + fn1)
    prec2 = tp2 / (tp2 + fp2)
    rec2 = tp2 / (tp2 + fn2)
    F1_1 = 2 * ((prec1 * rec1) / (prec1 + rec1))
    F1_2 = 2 * ((prec2 * rec2) / (prec2 + rec2))
    print("-------pos/neg--------")
    print("true pos: {} | false pos: {} | false neg: {}".format(tp1, fp1, fn1))
    print("precision: {} | recall: {} | F1: {}".format(prec1, rec1, F1_1))
    print("-------truth/dec-------")
    print("precision: {} | recall: {} | F1: {}".format(prec2, rec2, F1_2))


if __name__=='__main__':
    result1, result2 = [], []
    input_path = sys.argv[-1]
    # input_path = str(sys.argv[-1])
    
    X_train, y_train_pos, y_train_truth, X_test, y_test_pos, y_test_truth = load_data()

    stopwords = get_stopwords(X_train)
    vocab = get_vocab(X_train)
    # print(stopwords)
    
    word_counts, log_priors, label_count = train(X_train, y_train_pos, y_train_truth)
    
    # print(sorted(word_counts['neg'].items(), key=lambda k_v: k_v[1]))

    # log_priors, loglikelihood, vocab, doc_count, num_positives, num_truthful, num_negatives, num_deceptive = train(X_train, y_train_pos, y_train_truth)
    # print(log_priors)
    result1, result2 = classify(X_test, word_counts, log_priors)

    pos_accuracy = sum(1 for i in range(len(result1)) if result1[i] == y_test_pos[i]) / float(len(result1))
    print(pos_accuracy)

    truth_acc = sum(1 for i in range(len(result2)) if result2[i] == y_test_truth[i]) / float(len(result2))
    print(truth_acc)

    score()

