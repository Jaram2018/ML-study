from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import keras
import os
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

# define problem properties
# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(107, 200)))
model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(250):
    # generate new random sequence
    X,y = get_sequence()
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=20, verbose=2)
# evaluate LSTM
X,y = get_sequence()
yhat = model.predict_classes(X, verbose=0)
for i in y:
    print('Expected:', y[0, i], 'Predicted', yhat[0, i])


# class Regression():
#     """
#     영화리뷰 예측을 위한 Regression 모델입니다.
#     """
#     def __init__(self, embedding_dim: int, max_length: int):
#         """
#         initializer
#
#         :param embedding_dim: 데이터 임베딩의 크기입니다
#         :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
#         """
#         super(Regression, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.character_size = 251
#         self.output_dim = 1  # Regression
#         self.max_length = max_length
#
#         self.model = Sequential()
#         self.model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(self.max_length * self.embedding_dim, 200)))
#         self.model.add(TimeDistributed(Dense(1, activation='sigmoid')))
#         self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
#
#     def forward(self, data: list):
#         """
#
#         :param data: 실제 입력값
#         :return:
#         """
#         # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
#         batch_size = len(data)
#         # list로 받은 데이터를 torch Variable로 변환합니다.
#         data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
#         # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
#         if GPU_NUM:
#             data_in_torch = data_in_torch.cuda()
#         # 뉴럴네트워크를 지나 결과를 출력합니다.
#         embeds = self.embeddings(data_in_torch)
#         hidden = self.fc1(embeds.view(batch_size, -1))
#         # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
#         output = torch.sigmoid(self.fc2(hidden)) * 9 + 1
#         return output
