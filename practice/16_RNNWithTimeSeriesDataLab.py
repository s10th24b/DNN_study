import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Many-to-One
# 타임시리즈 데이터인 주식으로 실습
# 입력의 디멘션? 히든사이즈? 시퀀스 길이?
# 인풋 디멘션은 5개. 시퀀스는 7. 7일동안의 데이터 보니까.  아웃풋 디멘션(히든사이즈)은 1.

def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


time_steps = seq_length = 7
data_dim = 5
output_dim = 1
# Open,High,Low,Close,Volume
xy = np.loadtxt('data-02-stock_daily.csv',delimiter=',')
xy = xy[::-1] # reverse order (chronically ordered)
# ::-1 는 거꾸로 뒤집기
xy = MinMaxScaler(xy) # Normalizing
x = xy
y = xy[:,[-1]] # Close as Label

dataX = []
dataY = []
for i in range(0,len(y) - seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length] # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split to train and testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]),np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]),np.array(dataY[train_size:len(dataY)])

# input placeholders
X = tf.placeholder(tf.float32, [None,seq_length,data_dim])
Y = tf.placeholder(tf.float32, [None,1])

hidden_dim = 32
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
# LSTM의 마지막 output에서 FC를 또 하나 붙일거기 때문에, FC의 input으로
# 들어가는 RNN의 outputs dimension인 hidden_dim은 내 맘대로 정하자
outputs,_states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
print("outputs.shape:",outputs.shape)
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1],output_dim, activation_fn=None)
# 이렇게 FC를 마지막에 넣었음.
# outputs[:,-1]은, 맨 마지막 아웃풋만 쓰겠다는 뜻.
# We use the last cell's outpu
# 여기서 output_dim은 1.

# 이제 loss를 정해야하는데 여기서는 sequence_loss가 아니다. 그냥 하나의 loss가
# 아닌 linear loss
loss = tf.reduce_sum(tf.square(Y_pred - Y)) # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train,loss], feed_dict={X:trainX, Y:trainY})
    # print("i: {0}, l: {1}".format(i, l))
testPredict = sess.run(Y_pred, feed_dict={X: testX})

import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(testPredict)
# print("testY:",testY)
print("testY.shape:",testY.shape)
print("testPredict.shape:",testPredict.shape)
plt.show()
