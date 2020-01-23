import tensorflow as tf
import numpy as np
import pdb

# 우린 HiHelloRNNLab에서 Manually하게 one-hot으로 data creation을 했다.
# 각각 문자에 대한 인덱스와 해당되는 번호를 손으로 만들었다.
# 하지만 문자열이 길어지면 힘들다. 그래서 자동으로 해보자.

sample = "if you want you"
idx2char = list(set(sample))  # index -> char
print("idx2char:", idx2char)
char2idx = {c: i for i, c in enumerate(idx2char)}  # chat -> idx
print("char2idx:", char2idx)

sample_idx = [char2idx[c] for c in sample]
print("sample_idx:", sample_idx)
x_data = [sample_idx[:-1]]  # X data sample (0~n-1) hello: hell
y_data = [sample_idx[1:]]  # Y data sample (1~n) hello: ello
# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc..)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # X에서 끝에서 마지막 1개까지만 하니까
X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

num_classes = len(char2idx)
# 전체 몇개의 one-hot으로 만들어줄지를 정하는 num_classes
X_one_hot = tf.one_hot(X, num_classes)
# one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
# 한가지 주의할게 one-hot으로 만들때는 shape이 어떻게 변하는지를 살펴봐라
# printing by khj
sess = tf.Session()
print("x_data:", x_data)
print("y_data:", y_data)
print("X_one_hot:", sess.run(X_one_hot, feed_dict={X: x_data}))
print("X_max:", sess.run(tf.argmax(X_one_hot, -1), feed_dict={X: x_data}))  # (= x_data)
# printing by khj ###

cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)  # output size = one-hot size
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
# cell, X_one_hot,initial_state,dtype)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(sequence_loss)
prediction = tf.argmax(outputs, axis=-1)

# 이제 학습해야지.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # for i in range(1000):
    for i in range(10):
        l, _ = sess.run([loss, optimizer], feed_dict={X: x_data, Y: y_data})
        # print("loss:", l)
        result, outputs_res = sess.run([prediction, outputs], feed_dict={X: x_data})
        print("outputs_res:", outputs_res)
        print("outputs_res.shape:", outputs_res.shape)
        print("output_res -> result:", result)
        print("np.squeeze(result):", np.squeeze(result))
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("result_str:", result_str)

# 이건 잘 된다.

