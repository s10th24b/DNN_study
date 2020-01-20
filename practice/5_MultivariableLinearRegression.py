import tensorflow as tf
import numpy as np

# H(x) (Hypothesis) = W (weight) * x + b (bias)


# 만약 변수가 3개라면?
# H(x1,x2,x3) = w1x1 + w2x2 + w3x3 + b
# Matrix = w1x1 + w2x2 + w3x3 + ... + wnxn
# 변수가 많아지면 길게 계속 늘어뜨리기 번거롭다.
# 그러므로, Matrix를 사용한다.
# [ x1 x2 x3 .... xn] x [w1 w2 w3 ... wn]^(T) = (x1w1 + x2w2 + ... + xnwn)
# Matrix 를 쓸때는 x를 앞에다가 쓴다. 
# H(X) = X*W
# X와 W가 대문자라는 건 Matrix라는 암시.

# row가 하나의 인스턴스
# ( x11 x12 x13 )          = ( x11w1 + x12w2 + x13w3 )
# ( x21 x22 x23 ) * ( w1 ) = ( x21w1 + x22w2 + x23w3 )
# ( x31 x32 x33 ) * ( w2 ) = ( x31w1 + x32w2 + x33w3 )
# ( x41 x42 x43 ) * ( w3 ) = ( x41w1 + x42w2 + x43w3 )
# ( x51 x52 x53 )          = ( x51w1 + x52w2 + x53w3 )
#    [ 5 , 3]  *   [ 3 , 1 ]     =    [ 5 , 1 ]
#     H(X) = XW
# W의 크기는 어떻게 결정? -> x의 요소개수인 3받아서 y의 요소개수인 1
# n 은 인스턴스 개수
#    [ n , x]  *   [ x , y ]     =    [ n , y ]

# Lecture(theory):
    # H(x) = Wx + b

# Implementation(TensorFlow):
    # H(x) = XW 

x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]

# 만일 x가 100개가 넘어가면 힘들다. 그러므로 이제 이 표현은 안하고, 매트릭스를 사용할 것이다.

# placeholders for a tensor taht will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]),name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]),name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]),name = 'weight3')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize. Need a very small learning rate for this data set
optimizer =  tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session.
sess = tf.Session()
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train],feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,Y:y_data})

    if step % 10 == 0:
        print("step:",step, "Cost:",cost_val, "\nPrediction:",hy_val)
