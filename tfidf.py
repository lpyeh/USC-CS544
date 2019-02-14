#!/usr/bin/python

import math
from collections import *
from preprocess import *

def tf(doc):
    tf = defaultdict(float)
    length = len(doc)
    tf = defaultdict(int)

    # tf
    doc_counts = Counter(doc)
    # doc_counts = Counter(doc)
    for word, val in doc_counts.items():
        # print(word)
        if val > 0:
            tf[word] += 1.0
    return tf


def idf(X, vocab):
    total_docs = len(X)
    idf = defaultdict(float)
    doc_frequencies = defaultdict(int)

    for doc in X:
        counts = Counter(tokenize(doc))
        for word in counts.keys():
            doc_frequencies[word] += 1

    for word, val in doc_frequencies.items():
        idf[word] = math.log10(total_docs / float(val))

    return idf


def tfidf(doc, X, vocab):
    tfidf = defaultdict(float)
    tf_vec = tf(doc)
    idf_vec = idf(X, vocab)
    for word in doc:
        tfidf[word] = tf_vec[word] * idf_vec[word]
    print(tfidf)
    return tfidf


