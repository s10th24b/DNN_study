import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

nb_classes = 10 # 0~ 9
X = tf.placeholder(tf.float32,[None,784]) # nb of pixels = 784
Y = tf.placeholder(tf.float32,[None,nb_classes])

W = tf.Variable(tf.random.normal([784,nb_classes])) # W의 크기는 x크기 * y크기이다.
b = tf.Variable(tf.random.normal([nb_classes])) #b의 크기는 y의 크기와 같다.

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)
cost_i = -tf.reduce_sum(Y*tf.log(hypothesis),1)
cost = tf.reduce_mean(cost_i)
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs): #전체 데이터셋을 한번 돈거 = 1 epoch
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) # 1 epoch을 여러개로 나눈게 1 batch_size
        # Iterations...
        # 만약 1000개의 훈련 셋이 있고 배치사이즈가 500이면 2 iterations
        # print("mnist.train.num_examples:",mnist.train.num_examples)
        # print("total_batch:",total_batch)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c,_ = sess.run([cost,optimizer],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost+= c / total_batch
        # print("Epoch:",epoch+1,"cost:",avg_cost)
        print("Epoch:","%04d"%(epoch+1),"cost:","{:.9f}".format(avg_cost))
        # print("hypothesis:",sess.run(hypothesis,feed_dict={X:batch_xs,Y:batch_ys}))

    print("Testing...")
    print("Accuracy:",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
    # print("Accuracy:",accuracy.eval(session=sess,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))

    import matplotlib.pyplot as plt
    import random

    # get one and predict
    r = random.randint(0,mnist.test.num_examples -1)
    print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction:",sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
    plt.show()
