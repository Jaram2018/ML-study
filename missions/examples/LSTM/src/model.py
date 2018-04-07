from random import random

import torch
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.dataset import MovieReviewDataset

# create a sequence classification instance
DATASET_PATH = "../../movie-review/sample_data/movie_review/"
def get_sequence():
    dataset = MovieReviewDataset(DATASET_PATH, 200)
    X = array(dataset.reviews)
    y = array(dataset.labels)
    X = X.reshape(1, 107, 200)
    y = y.reshape(1, 107,1)
    return X, y

class Regression(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int, vocab_size, target_size, hidden_dim, num_layers=1, dropout=0.9):
        super(Regression, self).__init__()

        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # set up modules for recurrent neural networks
        self.rnn = nn.LSTM(embedding_dim=embedding_dim,
                           hidden_dim = hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=True)
        self.hideen2tag = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1,1, self.hidden_dim)), Variable(torch.zeros(1,1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        print(tag_scores)
        return (tag_scores * 9) + 1