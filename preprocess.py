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


def get_stopwords(text):
    stopwords = []
    data = {}
    for words in text:
        for k, v in Counter(tokenize(words)).items():
            if k not in data:
                data[k] = 0
            data[k] += v
    stopwords = [k for k, v in data.items() if v > 250 or len(k) <= 2]
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


            
        



