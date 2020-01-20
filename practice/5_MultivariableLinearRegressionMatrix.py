import tensorflow as tf
import numpy as np

# H(x) (Hypothesis) = W (weight) * x + b (bias)

# row가 하나의 인스턴스
# ( x11 x12 x13 )          = ( x11w1 + x12w2 + x13w3 )
# ( x21 x22 x23 ) * ( w1 ) = ( x21w1 + x22w2 + x23w3 )
# ( x31 x32 x33 ) * ( w2 ) = ( x31w1 + x32w2 + x33w3 )
# ( x41 x42 x43 ) * ( w3 ) = ( x41w1 + x42w2 + x43w3 )
# ( x51 x52 x53 )          = ( x51w1 + x52w2 + x53w3 )
#    [ 5 , 3]  *   [ 3 , 1 ]     =    [ 5 , 1 ]
#     H(X) = XW
# W의 크기는 어떻게 결정? -> x의 개수인 3받아서 y의 개수인 1
# n 은 인스턴스 개수
#    [ n , x]  *   [ x , y ]     =    [ n , y ]

# Lecture(theory):
    # H(x) = Wx + b

# Implementation(TensorFlow):
    # H(x) = XW 

#              (w1)
# (x1 x2 x3) * (w2)     =   (x1w1 + x2w2 + x3w3)
#              (w3)

# 인스턴스는 총 5개
# 인스턴스마다 묶음.
x_data = [[73.,80.,75.], [93.,88.,93.], [89.,91.,90.], [96.,98.,100.],[73.,66.,70.]]
y_data = [[152.],[185.],[180.],[196.],[142.]]


# placeholders for a tensor taht will be always fed.
X = tf.placeholder(tf.float32,shape=[None, 3]) #None은 정해지지 않음. 원하는 만큼 줄 수 있다. 각 엘리먼트는 3개를 가진다
Y = tf.placeholder(tf.float32, shape = [None,1])

# W의 크기는 어떻게 결정? -> x의 요소개수인 3받고 y의 요소개수인 1 받는다
W = tf.Variable(tf.random_normal([3,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

# hypothesis = x1*w1 + x2*w2 + x3*w3 + b
hypothesis = tf.matmul(X,W) + b

#Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize.
optimizer =  tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session.
sess = tf.Session()
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if step % 10 == 0:
        print("step:",step, "Cost:",cost_val, "\nPrediction:",hy_val)
