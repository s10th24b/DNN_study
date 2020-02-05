# 이전 15강 LongLong에서는 RNN이 하나밖에 없었다. 그래서 정확도가 낮았음
# 이제는 RNN을 쌓아서.
# Cell을 추가. MultiRNNCell
import tensorflow as tf
import numpy as np
import pdb
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
######################
cell = tf.contrib.rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 깊게 팔수있으.
# 그리고 RNN에서도 CNN에서처럼 소프트맥스를 마지막에 붙여주자
# softmax에 넣으려면 크기에 맞게 reshape
# X_for_softmax = tf.reshape(outputs,[-1, hidden_size])
# 어려워보이나, 그냥 기계적으로 이렇게 한다고만 알면 됨.
# 이렇게 하면 output들마다 크기가 hidden_size로 되고 Stack처럼 쌓여
# softmax(??...  softmax취하지도 않는데 왜 softmax layer라고 이름붙인거지 여기선..?)의 입력으로 들어간다.
# 그냥 fully-connected-layer라고 하는게 좋을듯.
# 그럼 이제 softmax의 output이 나온다. 그걸 어떻게 펼쳐줘야하나?
# outputs = tf.reshape(outputs,[batch_size,seq_length,num_classes])
######################
initial_state = cell.zero_state(batch_size,tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell,X_one_hot,initial_state=initial_state,dtype=tf.float32)
X_for_softmax = tf.reshape(outputs,[-1, hidden_size])
softmax_w = tf.get_variable("softmax_w",[hidden_size,num_classes])
softmax_b = tf.get_variable("softmax_b",[num_classes])
outputs = tf.matmul(X_for_softmax,softmax_w) + softmax_b # (= softmax output)
outputs = tf.reshape(outputs,[batch_size,seq_length,num_classes]) # softmax output을 펼쳐준다

weights = tf.ones([batch_size,seq_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weights)
# 사실, 저 logit에는 activation 함수를 거치지 않은 값을 넣어야한다. 그래야 좋은성능.
# RNN에서 output된 값은 이미 activation을 거친 값이기에 logit으로 넣는데에 불안정하다
# 그래서 그 값을 reshape해주고 WX+b의 형태인 affine sum으로 다시 정리를 해준후
# logit으로 주어야 학습이 잘된다
mean_loss = tf.reduce_mean(seq_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

prediction = tf.argmax(outputs,axis=-1)

from os import system
system('clear')
res= []
# 이제 학습해야지.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _, result = sess.run([mean_loss, optimizer,outputs], feed_dict={X:dataX, Y:dataY})
        # print("loss:", l)
        result, outputs_res = sess.run([prediction, outputs], feed_dict={X: dataX})
        # print("outputs_res:", outputs_res)
        # print("outputs_res.shape:", outputs_res.shape)
        # print("output_res -> result:", result)
        # print("result.shape:", result.shape)
        # print("np.squeeze(result):", np.squeeze(result))
        # print("np.squeeze(result).shape:", np.squeeze(result).shape)

        result = result[0]
        # print("result:", result)
        # print("np.squeeze(result):", np.squeeze(result))
        # print("np.squeeze(result).shape:", np.squeeze(result).shape)

        # for c in np.squeeze(result):
            # print(c)
        # print("np.squeeze(result).shape:", np.squeeze(result).shape)
        result_str = [idx2char[c] for c in np.squeeze(result)]
        origin_str = [idx2char[c] for c in np.squeeze(dataY[0])]
        # sys.stdout.flush()
        sys.stdout.write("origin_str:{0} \n\rresult_str:{1}\r".format(origin_str,result_str))
        # print("origin_str:",origin_str)
        # print("result_str:",result_str)
        res = result_str

print("final result:",res)
# 잘 안된다... 왜? -> 1. logits에서 매끄럽지가 않다. NN이 깊지가 않다.
