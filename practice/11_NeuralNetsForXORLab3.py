import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)

# XOR with "Wide" Neural Net

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random.normal(([2,10]),name='weight1'))
b1 = tf.Variable(tf.random.normal(([10]),name='bias1'))
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)
# 이런 걸 Wide하다고 한다.
# 여기서 중요한게, weight의 크기를 잘 정해줘야한다.
# 첫번째 레이어의 결과인 y가 두번째 레이어의 input으로 들어가므로, 두번째 input의 수와 같은 10로 해줘야한다.
W2 = tf.Variable(tf.random.normal(([10,1]),name='weight2'))
b2 = tf.Variable(tf.random.normal(([1]),name='bias2'))
hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
        if step % 100 == 0:
            print("step:",step,"cost:",sess.run(cost,feed_dict={X:x_data,Y:y_data}))
    h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
    print("hypothesis:",h)
    print("is_correct:",c)
    print("accuracy:",a)

    #result: good. accuracy:100%
