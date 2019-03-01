import json
import sys, os
import glob
from collections import defaultdict
import numpy as np
from preprocess import *



def load_data(path):
    # print("Loading Data")
    all_files = glob.glob(os.path.join(path, '*/*/*/*.txt'))
    test = []
    paths = []

    for file_ in all_files:
        class1, class2, fold, fname = file_.split('/')[-4:]
        text = open(file_).read()
        test.append(tokenize(clean(text)))
        
        paths.append(file_)
    return test, paths


def vectorize(X, vocab, method):
    # print("Vectorizing Data")
    if method == 'count':
        data = word_count(X, vocab)
    elif method == 'tfidf':
        data = np.array(tfidf(X, vocab), dtype=np.float32)
    return data


def load_params(textfile):
    positive_weights = defaultdict(float)
    truth_weights = defaultdict(float)
    
    with open(textfile, 'r') as in_:
        data = json.load(in_)
    
    positive_weights = np.array(data['positive'], dtype=np.float32)
    truth_weights = np.array(data['truthful'], dtype=np.float32)
    positive_bias = float(data['pos_bias'])
    truthful_bias = float(data['truth_bias'])
    vocab = data['vocab']

    return positive_weights, truth_weights, positive_bias, truthful_bias, vocab



def convert(x):
    return {k:float(v) for k,v in x.items()}



def classify(X_test, weights, bias):
    result = []
    for x in X_test:
        activation = np.sum(np.multiply(x, weights)) + bias
        if activation > 0:
            result.append(1)
        else:
            result.append(-1)
    return result



def write(positive_result, truthful_result, paths):
    out = open('percepoutput.txt', 'w')
    for pos, truth, path in zip(positive_result, truthful_result, paths):
        if truth == 1:
            out.write("truthful ")
        else:
            out.write("deceptive ")
        if pos == 1:
            out.write("positive ")
        else:
            out.write("negative ")
        out.write(path)
        out.write('\n')
    out.close()



if __name__=='__main__':
    input_path = sys.argv[-1]
    model_path = sys.argv[-2]

    
    pos_weights, truth_weights, pos_bias, truth_bias, vocab = load_params(model_path)
    X_test, paths = load_data(input_path)
    
    data = vectorize(X_test, vocab, method = 'count')

    pos_result = classify(data, pos_weights, pos_bias)
    truth_result = classify(data, truth_weights, truth_bias)
    write(pos_result, truth_result, paths)
