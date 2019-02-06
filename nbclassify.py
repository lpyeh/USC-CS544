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
from preprocess import *

def load_data():
    all_files = glob.glob(os.path.join(input_path, '*/*/*/*.txt'))
    test = []
    paths = []
    for file_ in all_files:
        class1, class2, fold, fname = file_.split('/')[-4:]
        text = open(file_).read()
        test.append(text)
        paths.append(file_)
    test = clean(test)
    return test, paths


def load_params():
    log_priors = defaultdict(float)
    loglikelihood = defaultdict(dict)
    # param_path = os.path.join(os.getcwd(), 'nbmodel.txt')
    with open('nbmodel.txt', 'r') as in_:
        data = in_.readlines()
    log_priors['pos'], log_priors['neg'], log_priors['truth'], log_priors['deceptive'] = [float(x) for x in data[2].split(',')]
    for i in range(5, len(data)):
        item = data[i].split(',')
        word = item[0]
        if word not in loglikelihood:
            loglikelihood['pos'][word] = 0.0
            loglikelihood['neg'][word] = 0.0
            loglikelihood['truth'][word] = 0.0
            loglikelihood['deceptive'][word] = 0.0
        loglikelihood['pos'][word] = float(item[1])
        loglikelihood['neg'][word] = float(item[2])
        loglikelihood['truth'][word] = float(item[3])
        loglikelihood['deceptive'][word] = float(item[4].strip('\n'))
    return loglikelihood, log_priors

    

def classify(X, path_list):
    outfile = open('nboutput.txt', 'w')
    
    pos_score, neg_score, truth_score, dec_score = 0.0, 0.0, 0.0, 0.0
    truth_result = []
    pos_result = []
    for x, path in zip(X, path_list):
        counts = word_count(tokenize(x))

        for word in counts.keys():
            # print(word)
            if word in stopwords:
                continue;
            """
            log_pos = math.log10((word_counts['pos'].get(word, 0.0) + 1) / (label_count['pos'] + len(vocab)))
            log_neg = math.log10((word_counts['neg'].get(word, 0.0) + 1) / (label_count['neg'] + len(vocab)))
            log_truth = math.log10((word_counts['truth'].get(word, 0.0) + 1) / (label_count['truth'] + len(vocab)))
            log_dec = math.log10((word_counts['deceptive'].get(word, 0.0) + 1) / (label_count['deceptive'] + len(vocab)))
            # print(word)
            """
            smooth = math.log(1 / 240 + len(loglikelihood['pos']))
            pos_score += loglikelihood['pos'].get(word, smooth)
            neg_score += loglikelihood['neg'].get(word, smooth)
            truth_score += loglikelihood['truth'].get(word, smooth)
            dec_score += loglikelihood['deceptive'].get(word, smooth)

        pos_score += log_priors['pos']
        neg_score += log_priors['neg']
        truth_score += log_priors['truth']
        dec_score += log_priors['deceptive']

        if neg_score > pos_score:
            outfile.write('negative ' )
        else:
            outfile.write('positive ')

        if dec_score > truth_score:
            outfile.write('deceptive ')
        else:
            outfile.write('truthful ')

        outfile.write(path)
        outfile.write('\n')

    outfile.close()
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
    input_path = sys.argv[-1]
    loglikelihood, log_priors = load_params()
    X_test, paths = load_data()
    stopwords = get_stopwords(X_test)
    result1, result2 = classify(X_test, paths)
