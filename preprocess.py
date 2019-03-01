from collections import Counter
import re


def clean(doc):
    doc = re.sub(r'[^A-Za-z\s\']', ' ', doc).lower()
    doc = re.sub(r'\n', ' ', doc)
    doc = re.sub(r'(.)\1+', r'\1\1', doc)
    return doc

def tokenize(text):
    text = text.split(" ")
    return [x.strip() for x in text]

"""
def word_count(text):
    return Counter(text)
"""

def word_count(docs, vocabulary):
    vector = []
    for doc in docs:
        doc_words = []
        count = Counter(doc)
        for word in vocabulary:
            if word in count:
                doc_words.append(count[word])
            else:
                doc_words.append(0.0)
        vector.append(doc_words)
    return vector


def get_vocab(docs, vocab_length):
    stopwords = ["she", "and", "him", "they", "then", "you", "the", "for", "that", "was", "not", "with", "had", "but", "had"]
    vocab = {}
    for doc in docs:
        count = Counter(doc)
        for word in count:
            if len(word) <= 2 or word in stopwords:
                continue;
            if word in vocab:
                if count[word] < 10:
                    continue
                else:
                    vocab[word] += count[word]
            else:
                vocab[word] = 0.0
                vocab[word] = count[word]
    vocab = sorted(vocab.items(), key=lambda k_v:-k_v[1])
    vocab = [x[0] for x in vocab][:vocab_length]
    return vocab


def get_stopwords(word_counts):
    stopwords = []
    for label, counts in word_counts.items():
        for k, v in counts.items():
            if v > 150:
                stopwords.append(k)
    stopwords.append("them")
    stopwords.append("then")
    stopwords.append("she")
    stopwords.append("he")
    stopwords.append("her")
    stopwords.append("him")
    stopwords.append("me")
    stopwords.append("they")
    stopwords.append("we")
    stopwords.append("their")
    stopwords.append("you")
    stopwords = list(set(stopwords))
    return stopwords


            
        



