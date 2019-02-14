#!usr/bin/python
import sys
import math
import os
from collections import *
import glob
from preprocess import *
import json

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
    word_counts = defaultdict(dict)
    with open('nbmodel.txt', 'r') as in_:
        data = json.load(in_)
    log_priors = data[0]
    label_counts = data[1]
    vocab = data[2]
    words = data[3]

    for k, v in words.items():
        word_counts[k] = convert(v)
    return convert(log_priors), convert(label_counts), convert(vocab), word_counts

def convert(x):
    return {k:float(v) for k,v in x.items()}

def classify(X, path_list):
    outfile = open('nboutput.txt', 'w')
    
    for x, path in zip(X, path_list):
        pos_score, neg_score, truth_score, dec_score = 0.0, 0.0, 0.0, 0.0
        counts = word_count(tokenize(x))

        for word, count in counts.items():

            if word in stopwords or len(word) <= 2 or word not in vocab:
                continue;
            
            prob_word = vocab[word] / sum(vocab.values())

            # Laplace smoothing
            likelihood_pos = (word_counts['pos'].get(word, 0.0) + 1) / (sum(word_counts['pos'].values()) + len(vocab))
            likelihood_neg = (word_counts['neg'].get(word, 0.0) + 1) / (sum(word_counts['neg'].values()) + len(vocab))
            likelihood_truth = (word_counts['truth'].get(word, 0.0) + 1) / (sum(word_counts['truth'].values()) + len(vocab))
            likelihood_dec = (word_counts['deceptive'].get(word, 0.0) + 1) / (sum(word_counts['deceptive'].values()) + len(vocab))
            
            pos_score += math.log(likelihood_pos * count / prob_word)
            neg_score += math.log(likelihood_neg * count / prob_word)
            truth_score += math.log(count * likelihood_truth / prob_word)
            dec_score += math.log(count * likelihood_dec / prob_word)
            
        pos_score = math.exp(pos_score + log_priors['pos'])
        neg_score = math.exp(neg_score + log_priors['neg'])
        truth_score = math.exp(truth_score + log_priors['truth'])
        dec_score = math.exp(dec_score + log_priors['deceptive'])

        outfile.write('deceptive ') if dec_score > truth_score else outfile.write('truthful ')
        outfile.write('negative ') if neg_score > pos_score else outfile.write('positive ')
        outfile.write(path)
        outfile.write('\n')
    outfile.close()

if __name__=='__main__':
    input_path = sys.argv[-1]
    log_priors, label_counts, vocab, word_counts = load_params()
    stopwords = get_stopwords(word_counts)
    X_test, paths = load_data()
    classify(X_test, paths)
