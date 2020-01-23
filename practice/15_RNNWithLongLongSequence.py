import tensorflow as tf
import numpy as np
import pdb
# 하지만, 만약에 엄~청나게 긴 문자열이라면?
sentence = ("if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them\
              to long for the endless immensity of the sea")
# 여러개의 배치를 준다.
# dataset
# 0 if you wan -> f you want
# 1 f you want ->  you want
# 2  you want  -> you want t
# 3 you want t -> ou want to
# ...
# 168 of the se -> of the sea
# 169 of the sea -> f the sea.


char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

idx2char = list(set(sentence))  # index -> char
print("idx2char:", idx2char)
char2idx = {c: i for i, c in enumerate(idx2char)}  # chat -> idx
print("char2idx:", char2idx)

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
seq_length = len(char_set)-1

print("char_set:", char_set)
print("char_dic:", char_dic)
dataX = []
dataY = []
for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i+seq_length]
    y_str = sentence[i+1:i+seq_length+1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str] # x str to index
    y = [char_dic[c] for c in y_str] # y str to index

    dataX.append(x)
    dataY.append(y)
batch_size = len(dataX)

X = tf.placeholder(tf.int32,[None,seq_length])
Y = tf.placeholder(tf.int32,[None,seq_length])
X_one_hot = tf.one_hot(X,num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
initial_state = cell.zero_state(batch_size,tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell,X_one_hot,initial_state=initial_state,dtype=tf.float32)

weights = tf.ones([batch_size,seq_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weights)
loss = tf.reduce_mean(seq_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs,axis=-1)

# 이제 학습해야지.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, optimizer], feed_dict={X:dataX, Y:dataY})
        print("loss:", l)
        result, outputs_res = sess.run([prediction, outputs], feed_dict={X: dataX})
        print("outputs_res:", outputs_res)
        print("outputs_res.shape:", outputs_res.shape)
        print("output_res -> result:", result)
        print("np.squeeze(result):", np.squeeze(result))
        print("np.squeeze(result).shape:", np.squeeze(result).shape)
        result = result[0]
        print("result:", result)
        print("np.squeeze(result):", np.squeeze(result))
        print("np.squeeze(result).shape:", np.squeeze(result).shape)
        # for c in np.squeeze(result):
            # print(c)
        # print("np.squeeze(result).shape:", np.squeeze(result).shape)
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("result_str:", result_str)

# 잘 안된다... 왜? -> 1. logits에서 매끄럽지가 않다. NN이 깊지가 않다.
