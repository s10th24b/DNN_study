import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

x_train = [[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
scaled_x_train = MinMaxScaler(x_train)
print("scaled_x_train:",scaled_x_train)
y_train = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
scaled_y_train = MinMaxScaler(y_train)
print("scaled_y_train:",scaled_y_train)

x_test = [[2,1,1],[3,1,2],[3,3,4]]
scaled_x_test = MinMaxScaler(x_test)
y_test = [[0,0,1],[0,0,1],[0,0,1]]
scaled_y_test = MinMaxScaler(y_test)

X = tf.placeholder(tf.float32,[None,3])
Y = tf.placeholder(tf.float32,[None,3])
W = tf.Variable(tf.random.normal([3,3]))
b = tf.Variable(tf.random.normal([3]))

learning_rate = 0.01

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)
# cost_i = -tf.reduce_sum(Y*tf.log(hypothesis),axis=1)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis,axis=1)
is_correct = tf.equal(prediction,tf.argmax(Y,axis=1))
acc = tf.reduce_mean(tf.cast(is_correct,dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # for step in range(500):
        # sess.run(optimizer,feed_dict={X:x_train,Y:y_train})

        # print("step:",step,"cost:",sess.run([cost,prediction],feed_dict={X:x_train,Y:y_train}))
        # print("step:",step,"acc:",sess.run([acc,tf.argmax(y_train,1)],feed_dict={X:x_train,Y:y_train}))
    # print("Testing....")
    # print("acc:",sess.run(acc,feed_dict={X:x_test,Y:y_test}))

    for step in range(500):
        sess.run(optimizer,feed_dict={X:x_train,Y:y_train})

        print("step:",step,"cost:",sess.run([cost,prediction],feed_dict={X:scaled_x_train,Y:scaled_y_train}))
        print("step:",step,"acc:",sess.run([acc,tf.argmax(scaled_y_train,1)],feed_dict={X:scaled_x_train,Y:scaled_y_train}))
    print("Testing....")
    print("acc:",sess.run(acc,feed_dict={X:scaled_x_test,Y:scaled_y_test}))
