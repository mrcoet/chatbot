import torch.nn as nn
import nltk
import string
from nltk.stem.porter import PorterStemmer
import numpy as np


class NerualNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NerualNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


def tokenize(sentence):
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    return nltk.word_tokenize(sentence)


stemmer = PorterStemmer()


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idex, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idex] = 1.0
    return bag
