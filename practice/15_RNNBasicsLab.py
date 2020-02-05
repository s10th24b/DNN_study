# Neural Net중 가장 활용도가 높다고 알려진 RNN에 대해서 실습.
import tensorflow as tf
import numpy as np

# TF를 이용하면 아주 쉽게 구현이 가능하다.
# 첫번째로 cell 이라는 것을 만든다.
# 만들때 가장 중요한건, 출력의 크기를 정해준다.= num_units
# cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
# ...
# 두번째 단계에서는 셀을 실제 구동시키고 아웃풋을 뽑아내야한다.
# 우리가 만든 cell을 넣고, 우리가 원하는 입력 데이터를 넣는다.
# outputs, _states = tf.nn.dynamic_rnn(cell,x_data,dtype=tf.float32)
# 그럼 이 dynamic_rnn은 2가지 출력을 낸다. 하나는 아웃풋과 하나는 마지막 state 값
# state값은 우리가 직접 사용할 일은 많지 않고 주로 이 아웃풋을 사용하게 된다.
# 이렇게 나눈 이유는 셀을 생성하는 부분과 셀을 학습시키는 부분을 나누려고 한 것.
# 예를 들어 LSTM으로 바꾸고 싶으면 학습부분은 그대로 두고,
# cell = tf.contrib.rnn.BasicLSTMcell(num_units=hidden_size)
# 로 바꾸면 된다!

#############################################3
sess = tf.Session()
hidden_size = 2
# hidden_size는 출력의 크기. 일단 2로 해보자.
sequence_length = 5
# sequence_length는 몇개의 입력이 차례로 들어갈건지, 즉 셀을 몇개 '더' 펼칠건지. 일단 5로 해보자. hello
# 굳이 변수로 선언 안해도, 우리가 입력을 넣을때 정해진다.
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)

x_data = np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]],dtype=np.float32) #hello
# input dimension: (1,5,4) output_dimension: (1,5,2)
outputs, _states = tf.nn.dynamic_rnn(cell,x_data,dtype=tf.float32)

sess.run(tf.global_variables_initializer())
print("x_data:",x_data)
print("x_data.shape:",x_data.shape)
print("outputs:",sess.run(outputs))
print("outputs.shape:",outputs.shape)
print("_states:",sess.run(_states))

# 이제 마지막 단계. 우리가 이렇게 문자 하나씩 학습시키면  얼마나 느릴까.
# 이걸 효율적으로 하는건 문자열을 여러개, 즉 어려운말로 batch_size로 데이터를 여러개 줄 수 있다.
batch_size = 3
# 이것도 굳이 선언안해도 입력에서 정해짐.
# x_data = np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]], #hello
                    # [[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0]], #eolll
                    # [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,1,0,0],[0,0,1,0]]], #lleel
                    # dtype=np.float32)
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

x_data = np.array([[h,e,l,l,o], # hello
                   [e,o,l,l,l], # eolll
                   [l,l,e,e,l]], # lleel
                  dtype=np.float32)
# input dimension: (3,5,4) output_dimension: (3,5,2)

outputs, _states = tf.nn.dynamic_rnn(cell,x_data,dtype=tf.float32)
sess.run(tf.global_variables_initializer())
print("x_data:",x_data)
print("outputs:",sess.run(outputs))
print("x_data.shape:",x_data.shape)
print("outputs.shape:",outputs.shape)
print("_states:",sess.run(_states))
print("_states.shape:",sess.run(_states).shape)
