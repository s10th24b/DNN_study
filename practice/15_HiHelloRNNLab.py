import tensorflow as tf
import numpy as np

#hihello를 훈련시켜볼 것이다.

# text: 'hihello'
# unique chars: h, i, e, l, o
# voc index: h:0, i:1, e:2, l:3, o:4


sess = tf.Session()
# Language Model은 그 다음 문자가 뭔지 맞춰야하니까, 입력과 같은 5를 출력 사이즈로 준다.
hidden_size = 5 #hielo
input_dim = 5 # one=hot size
sequence_length = 6 # ihello
batch_size = 1 #문자열 1개 넣음

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
# cell = tf.contrib.rnn.GRUCell(num_units=hidden_size)

# 이것에 기반해서 데이터를 만들자.
idx2char = ['h','i','e','l','o']
x_data = [[0,1,0,2,3,3]]
x_one_hot = [[[1,0,0,0,0], #h
[0,1,0,0,0],#i
[1,0,0,0,0],#h
[0,0,1,0,0],#e
[0,0,0,1,0],#l
[0,0,0,1,0]]]#l

y_data = [[1,0,2,3,3,4]] #ihello

# hihell -> ihello
X = tf.placeholder(tf.float32,[None,sequence_length,input_dim]) # X one-hot
Y = tf.placeholder(tf.int32,[None,sequence_length]) # Y Label

# input dimension: (1,6,5) output_dimension: (1,6)
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size) #hidden_size = 5


initial_state = cell.zero_state(batch_size,tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell,X,initial_state=initial_state,dtype=tf.float32)
# cell, X, initial_state, dtype

# 자 이렇게 rnn 신경망을 만들었는데, cost 함수가 필요하다.
# 우리가 지금까지 써왔던 softmax_cross_entropy_with_logits 를 써도 되긴하지만 복잡해지는 문제가 있다.
# 그래서 sequence_loss라는 함수를 쓰는데 아주 멋지다.

# outputs을 cost함수의 logits으로 사용할 것이다.
weights = tf.ones([batch_size,sequence_length])
#일단 가중치를 다 x데이터와 같은 크기의 1로 설정하고

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weights)
# 여기선 outputs을 logits으로 바로 넣고있는데 사실 좋은게 아니라 틀린거다.
# 간단하게 하기위해 일단은 이렇게 해보자.
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs,2) #왜 2?
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("x_data:",x_data)
    # print("outputs:",sess.run(outputs))
    # print("x_data.shape:",x_data.shape)
    # print("outputs.shape:",outputs.shape)
    # print("_states:",sess.run(_states))
    for i in range(2000):
        l,_ = sess.run([loss,train],feed_dict={X:x_one_hot,Y:y_data})
        result = sess.run(prediction,feed_dict={X:x_one_hot})
        print(i,"loss:",l,"prediction:",result,"result:","true Y:",y_data)

        # print prediction character using dictionary
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\t Prediction str:",''.join(result_str))

