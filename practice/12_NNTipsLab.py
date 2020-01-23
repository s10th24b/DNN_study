from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Use softmax, NN, Xavier, Deep NN

select = input("select option(All:0, softmax_cross_entropy_with_logits=1,NN=2,Xavier=3,Deep NN=4,Dropout=5): ")
select=int(select)
if select == 0 or select == 1:
    # Softmax classifier for MNIST
    print('Softmax classifier for MNIST')
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    W = tf.Variable(tf.random.normal([784,10]))
    b = tf.Variable(tf.random.normal([10]))
    learning_rate = 0.001
    batch_size = 100
    training_epochs = 50
    hypothesis = tf.matmul(X,W)+b
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs,Y:batch_ys}
            c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost +=c/total_batch

        print("Epoch:",(epoch+1),'cost:',avg_cost)
    print("Learning Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Cross_entropy Acc:",sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    # 90%

    r = random.randint(0,mnist.test.num_examples-1)
    plt.imshow(
            mnist.test.images[r : r + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
    print("answer:",sess.run(tf.argmax(mnist.test.labels[r : r + 1],-1)))
    plt.show()

if select == 0 or select == 2:
    # NN for MNIST
    print('NN for MNIST')
    X = tf.placeholder(tf.float32,[None,784])
    Y = tf.placeholder(tf.float32,[None,10])
    W1 = tf.Variable(tf.random.normal([784,256]))
    b1 = tf.Variable(tf.random.normal([256]))
    L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

    W2 = tf.Variable(tf.random.normal([256,256]))
    b2 = tf.Variable(tf.random.normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

    W3 = tf.Variable(tf.random.normal([256,10]))
    b3 = tf.Variable(tf.random.normal([10]))
    hypothesis = tf.matmul(L2,W3)+b3
    learning_rate = 0.001
    batch_size = 100
    training_epochs = 15

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs,Y:batch_ys}
            c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost +=c/total_batch

        print("Epoch:",(epoch+1),'cost:',avg_cost)
    print("Learning Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("NN Acc:",sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    # 94.5%

    r = random.randint(0,mnist.test.num_examples-1)
    plt.imshow(
            mnist.test.images[r : r + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
    print("answer:",sess.run(tf.argmax(mnist.test.labels[r : r + 1],-1)))
    plt.show()
if select == 0 or select == 3:
    # Xavier for MNIST
    print('Xavier for MNIST')
    X = tf.placeholder(tf.float32,[None,784])
    Y = tf.placeholder(tf.float32,[None,10])
    #xavier initialization
    W1 = tf.get_variable("W1",shape=[784,256],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random.normal([256]))
    L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

    W2 = tf.get_variable("W2",shape=[256,256],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random.normal([256]))
    L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

    W3 = tf.get_variable("W3",shape=[256,10],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random.normal([10]))
    hypothesis = tf.matmul(L2,W3)+b3
    learning_rate = 0.001
    batch_size = 100
    training_epochs = 15

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs,Y:batch_ys}
            c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost +=c/total_batch

        print("Epoch:",(epoch+1),'cost:',avg_cost)
    print("Learning Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Xavier Acc:",sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    # 97.8%

    r = random.randint(0,mnist.test.num_examples-1)
    plt.imshow(
            mnist.test.images[r : r + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
    print("answer:",sess.run(tf.argmax(mnist.test.labels[r : r + 1],-1)))
    plt.show()
if select == 0 or select == 4:
    # Deep NN for MNIST
    print('Deep NN for MNIST')
    X = tf.placeholder(tf.float32,[None,784])
    Y = tf.placeholder(tf.float32,[None,10])
    #xavier initialization
    W1 = tf.get_variable("W1",shape=[784,512],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random.normal([512]))
    L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

    W2 = tf.get_variable("W2",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random.normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

    W3 = tf.get_variable("W3",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random.normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)

    W4 = tf.get_variable("W4",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random.normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)

    W5 = tf.get_variable("W5",shape=[512,10],initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random.normal([10]))
    hypothesis = tf.matmul(L4,W5)+b5
    learning_rate = 0.001
    batch_size = 100
    training_epochs = 15

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs,Y:batch_ys}
            c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost +=c/total_batch

        print("Epoch:",(epoch+1),'cost:',avg_cost)
    print("Learning Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Deep NN Acc:",sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    # 97% #Lesser than shallower NN with xavier. Maybe by overfitting.
    # To prevent overfitting? -> Dropout! in next select.

    r = random.randint(0,mnist.test.num_examples-1)
    plt.imshow(
            mnist.test.images[r : r + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
    print("answer:",sess.run(tf.argmax(mnist.test.labels[r : r + 1],-1)))
    plt.show()
if select == 0 or select == 5:
    # Dropout for MNIST
    # Dropout cut the connection between layers, Preventing overfitting
    print('Dropout for MNIST')

    # dropout (keep_prob) rate 0,7 on training, but should be 1 for testing
    keep_prob = tf.placeholder(tf.float32)
    # How many nodes will you keep? : 0.7 : cut out 30% nodes

    X = tf.placeholder(tf.float32,[None,784])
    Y = tf.placeholder(tf.float32,[None,10])
    #xavier initialization
    W1 = tf.get_variable("W1",shape=[784,512],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random.normal([512]))
    L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
    L1 = tf.nn.dropout(L1,keep_prob=keep_prob)

    W2 = tf.get_variable("W2",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random.normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
    L2 = tf.nn.dropout(L2,keep_prob=keep_prob)

    W3 = tf.get_variable("W3",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random.normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
    L3 = tf.nn.dropout(L3,keep_prob=keep_prob)

    W4 = tf.get_variable("W4",shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random.normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
    L4 = tf.nn.dropout(L4,keep_prob=keep_prob)

    W5 = tf.get_variable("W5",shape=[512,10],initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random.normal([10]))
    hypothesis = tf.matmul(L4,W5)+b5
    learning_rate = 0.001
    batch_size = 100
    training_epochs = 15

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # make dropout rate to 0.7 in training
            feed_dict = {X:batch_xs,Y:batch_ys,keep_prob:0.7}
            c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost +=c/total_batch

        print("Epoch:",(epoch+1),'cost:',avg_cost)
    print("Learning Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # make dropout rate to 1 in testing
    print("Dropout Acc:",sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels,keep_prob:1}))
    # over 98% #Bigger than Deep NN which is without Dropout. Maybe by overfitting.

    r = random.randint(0,mnist.test.num_examples-1)
    plt.imshow(
            mnist.test.images[r : r + 1].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
        )
    print("answer:",sess.run(tf.argmax(mnist.test.labels[r : r + 1],-1)))
    plt.show()

