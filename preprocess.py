from collections import Counter
import re


def clean(text):
    cleaned = []
    for doc in text:
        doc = re.sub(r'[^A-Za-z\s\']', ' ', doc).lower()
        doc = re.sub(r'\n', ' ', doc)
        doc = re.sub(r'(.)\1+', r'\1\1', doc)
        cleaned.append(doc)
    return cleaned

def tokenize(text):
    text = text.split(" ")
    return [x.strip() for x in text]

def word_count(text):
    return Counter(text)


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


            
        



