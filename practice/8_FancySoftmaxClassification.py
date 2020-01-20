import tensorflow as tf
import numpy as np

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]] # one-hot encoding

X = tf.placeholder("float",[None,4])
Y = tf.placeholder("float",[None,3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

logits = tf.matmul(X,W) + b #logit = score
hypothesis = tf.nn.softmax(logits)

Y_label = tf.argmax(Y,axis=1) 
Y_one_hot = tf.one_hot(Y_label,nb_classes) # [[[ -> (?, 1, 3)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes]) # [[ -> (?, 3)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1)) # Y = one-hot
# 복잡하므로 간소화시킨다.
# softmax_cross_entropy_with_logits

#Cross entropy cost/loss
# hypothesis 가 아닌, logits을 넣는다! softmax 들어가기 전의 값
# cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})

        if step % 100 == 0:
            print("step:",step,"cost:",sess.run(cost,feed_dict={X:x_data,Y:y_data}))
            sess.run(hypothesis,feed_dict={X:x_data,Y:y_data})
    # print("Y_one_hot:",sess.run(Y_one_hot,))
    print("Y_label:",sess.run(Y_label,feed_dict={Y:y_data}))
    print("Y_label.shape:",tf.shape(sess.run(Y_label,feed_dict={Y:y_data})))
    print("Y_one_hot:",sess.run(Y_one_hot,feed_dict={Y:y_data}))
    print("Y_one_hot.shape:",tf.shape(sess.run(Y_one_hot,feed_dict={Y:y_data})))
    # print("tf.one_hot Y_one_hot:",sess.run(Y_one_hot))
    # print("reshaped Y_one_hot:",sess.run(Y_one_hot))

