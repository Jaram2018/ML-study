import tensorflow as tf
from tensorflow.contrib import rnn

seq_length = 5
rnn_size = 6
input_data = [[1,2,3,4,5],[6,1,8,9,10],[11,12,1,14,15]]

embedding = tf.get_variable("embedding", [20, rnn_size], initializer=tf.contrib.layers.xavier_initializer())
result = tf.nn.embedding_lookup(embedding, input_data)
inputs = tf.split(result, seq_length, 1)
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

cell_bw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell_fw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
bi = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,cell_fw.zero_state(3, tf.float32),cell_bw.zero_state(3, tf.float32),sequence_length=[5,5,5])

with tf.Session() as session:
    tf.initialize_all_variables().run()
    print(session.run(bi))