import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class RNN():
    def __init__(self,name,sess):
        self._name = name
        self._sess = sess

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,name):
        self._name = name

    def predict(self):
        pass


sess = tf.Session()
my_model1 = RNN('mymodel1',sess)


hidden_size = 32
input_dim = 1
output_dim = 1
sequence_length = 3
batch_size = 1

x_data = np.array([22,23,18,19,23,22,18],dtype=np.float32)
x_data = np.expand_dims(x_data,axis=-1)
x_data = np.expand_dims(x_data,axis=-1)
y_data = np.array([82.2,81.9,83.4,82.8,82.5,81.7,82.7],dtype=np.float32)
y_data = np.expand_dims(y_data,axis=-1)
print("x_data:",x_data)
print("y_data",y_data)
for i in range(0,len(x_data)-sequence_length):
    print(i)
    trainX = x_data[i:i+sequence_length+1]
    trainY = y_data[i:i+sequence_length+1]
    print(i+sequence_length)
X = tf.placeholder(tf.float32,[None,1,input_dim])
Y = tf.placeholder(tf.float32,[None,1])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 깊게 팔수있으.
outputs, _states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
print("outputs.shape:",outputs.shape)
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1],output_dim,activation_fn=None)
loss = tf.reduce_sum(tf.square(Y_pred-Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.global_variables_initializer())

for i in range(10000):
    _, l = sess.run([train,loss], feed_dict={X:trainX, Y:trainY})
    # print("i: {0}, l: {1}".format(i, l))
testPredict = sess.run(Y_pred, feed_dict={X: trainX})
# sess.run(tf.global_variables_initializer())
# print("x_data:",x_data)
# print("y_data",y_data)
import pdb; pdb.set_trace()  # XXX BREAKPOINT
# print("outputs:",sess.run(outputs,feed_dict={X:x_data}))
# print("_states:",_states)
plt.plot(y_data)
plt.plot(testPredict)
plt.show()
