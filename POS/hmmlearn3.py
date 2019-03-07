from collections import defaultdict
import json
import glob
import numpy as np
import os
from os.path import basename
from preprocess import *
import sys
# Need to output transmission and emission probs


class HMM():
    def __init__(self, path):
        self.train = []
        self.dev_raw = []
        self.dev_test = []
        self.data = []
        self.word_tags = defaultdict(dict)
        self.tag_tags = defaultdict(dict)
        self.transition_tag_counts = defaultdict(float)
        self.emission_tag_counts = defaultdict(float)
        self.initial_states = defaultdict(dict)
        self.path = path
        self.transitions = defaultdict(dict)
        self.emissions = defaultdict(dict)

    def load_data(self):
        files = glob.glob(os.path.join(self.path, '*.txt'))
        for file_ in files:
            if 'train_tagged' in file_:
                with open(file_) as f:
                    self.train.extend(f.readlines())
            

    def restructure(self):
        # make each line a list of tuples (word, tag)
        # new_data is a list of lists
        for line in self.train:
            line_data = []
            line = line.split(' ')
            for item in line:
                item = item.strip().rsplit('/', 1)
                line_data.append((item[0], item[1]))
            self.data.append(line_data)


    def get_counts(self):
        for line in self.data:
            for i in range(len(line)):
                word = line[i][0]
                tag = line[i][1]
                # if it's the first item in the sentence
                # add a start state
                if i == 0:
                    if tag not in self.tag_tags['q0']:
                        self.tag_tags['q0'][tag] = 0.0
                    self.tag_tags['q0'][tag] += 1.0
                # else if it's any of the items - the last item in sequence
                # update the count of the total tag counts, and update the tag -> next_tag count
                if i < len(line) - 1:
                    next_tag = line[i + 1][1]
                    if next_tag not in self.tag_tags[tag]:
                        self.tag_tags[tag][next_tag] = 0.0
                    self.transition_tag_counts[tag] += 1.0
                    self.tag_tags[tag][next_tag] += 1.0

                # else if it's the last item in the sequence
                # use end state
                elif i == len(line) - 1:
                    if 'end' not in self.tag_tags[tag]:
                        self.tag_tags[tag]['end'] = 0.0
                    self.tag_tags[tag]['end'] += 1.0

                if tag not in self.word_tags[word]:
                    self.word_tags[word][tag] = 0.0
                
                # update the emission tag counts (this will include all the items in the sequence)
                self.emission_tag_counts[tag] += 1.0
                # update word count
                self.word_tags[word][tag] += 1.0


    def get_transition_probs(self):
        for key, val in self.tag_tags.items():
            denominator = self.transition_tag_counts[key]
            for item in val:
                if key == 'q0':
                    if item not in self.initial_states[key]:
                        self.initial_states[key][item] = 0.0
                    self.initial_states[key][item] = self.tag_tags[key][item] / len(self.data)
                elif item == 'end':
                    continue;
                else:
                    if item not in self.transitions[key]:
                        self.transitions[key][item] = 0.0
                    self.transitions[key][item] = self.tag_tags[key][item] / denominator
    

    def get_emission_probs(self):
        for word, val in self.word_tags.items():
            for tag in val:
                if tag not in self.emissions[word]:
                    self.emissions[word][tag] = 0.0
                if self.emission_tag_counts[tag] == 0:
                    continue;
                self.emissions[word][tag] = self.word_tags[word][tag] / self.emission_tag_counts[tag]


    
    def write_file(self):
        write_dict = defaultdict(dict)
        write_dict['initial_states'] = self.initial_states
        write_dict['transition_probs'] = self.transitions
        write_dict['emission_probs'] = self.emissions
        with open('hmmmodel.txt', 'w', encoding='utf-8') as outfile:
            json.dump(write_dict, outfile, indent=2, ensure_ascii=False)


if __name__=='__main__':
    input_path = sys.argv[-1]
    model = HMM(input_path)
    model.load_data()
    model.restructure()
    model.get_counts()
    model.get_transition_probs()
    model.get_emission_probs()
    model.write_file()
    

            






                




