from collections import defaultdict
import json
import glob
import numpy as np
import os
from os.path import basename
from preprocess import *
import sys


class Decode():
    def __init__(self, path):
        self.path = path
        self.initial = np.array(0)
        self.transitions = np.array(0)
        self.emissions = np.array(0)
        self.data = defaultdict(dict)
        self.T1 = defaultdict(dict)
        self.T2 = defaultdict(dict)
        self.test = []
        self.test_w_tags = []
        self.results = []
        self.num_tags = 0


    def load_data(self, path):
        files = glob.glob(os.path.join(self.path, '*.txt'))
        for file_ in files:
            if 'dev_raw' in file_:
                with open(file_, encoding='utf-8') as fin:
                    self.test.extend(fin.readlines())
        
        with open('hmmmodel.txt', 'r', encoding='utf-8') as jin:
            self.data = json.load(jin)
        
        self.num_tags = len(self.transitions)
        # for item in data:
            # item = item.strip().split()
            # self.test.append(item)
        
            # TODO: recast all these items as float
    
    def convert_tuple(self):
        emission_tuples = []
        transition_tuples = []
        initial_tuples = []

        dt1 = {'names': ['tag1', 'tag2', 'prob'], 'formats': ["S3", "S3", np.float32]}
        dt2 = {'names': ['word', 'tag', 'prob'], 'formats': ["S20", "S3", np.float32]}

        for tag1, tags in self.data['emission_probs'].items():
            for tag2, prob in tags.items():
                emission_tuples.append((tag1, tag2, float(prob)))

        for tag1, tags in self.data['transition_probs'].items():
            for tag2, prob in tags.items():
                transition_tuples.append((tag1, tag2, float(prob)))

        for tag1, tags in self.data['initial_state'].items():
            for tag2, prob, in tags.items():
                initial_tuples.append((tag1, tag2, float(prob)))

        self.initial = np.array(initial_tuples, dtype=dt1)
        self.transitions = np.array(transition_tuples, dtype=dt1)
        self.emissions = np.array(emission_tuples, dtype=dt2)

    def viterbi(self, transitions, emissions):
        initial = transitions['tag1']['prob']

    # TODO: IMPLEMENT SMOOTHING
    def compute_probs(self):
        for line in self.test:
            line_dict = defaultdict(dict)
            for i in range(len(line)):
                if i == 0 or i == len(line) - 1:
                    word = line[i]
                if i < len(line) - 1:
                    prev_word = line[i]
                    word = line[i + 1]
                if word in self.emissions.keys():
                    for tag in self.emissions[word]:
                        # SMOOTHING
                        if i == 0:
                            line_dict[word][tag] = self.emissions[word][tag] * self.transitions['q0'][tag]
                        elif i < len(line) - 1:
                            probs = []
                            for prev_tag in line_dict[prev_word]:
                                # if self.transition[prev_tag][tag] == 0:
                                    # transition = 
                                probs.append(line_dict[prev_word][prev_tag] * self.emissions[word][tag] * transition)
                            line_dict[word][tag] = max(probs)
            self.viterbi.append(line_dict)
        print(self.viterbi)


    def get_tags(self):
        for i in range(len(self.test)):
            text = self.test[i]
            probabilities = self.viterbi[i]
            # for item in reversed(text):


if __name__=='__main__':
    input_path = sys.argv[-1]
    decode = Decode(input_path)
    decode.load_data(input_path)
    decode.convert_tuple()
    # decode.compute_probs()










