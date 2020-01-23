import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32,[None,784])
X_img = tf.reshape(X,[-1,28,28,1])  # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32,[None,10])



# L1 ImgIn shape = (?, 28, 28, 1)
W1 = tf.Variable(tf.random.normal([3,3,1,32],stddev=0.01)) #3x3의 필터, 들어온개수(맨 처음엔 색깔 1개니까 1), 출력개수(필터개수)
# Conv -> (?,28,28,32)
# Pool -> (?,14,14,32)
L1 = tf.nn.conv2d(X_img,W1,strides=[1,1,1,1],padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool2d(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L1 = tf.nn.dropout(L1,keep_prob=keep_prob)
# Conv2D: (?,28,28,32)
# Relu: (?,28,28,32)
# MaxPool: (?,14,14,32)
# 여기까지 거치면 (?, 14, 14, 32) 임.


# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random.normal([3,3,32,64],stddev=0.01)) #3x3의 필터, 들어온개수, 출력개수(필터개수)

# Conv -> (?,14,14,64)
# Pool -> (?,7,7,64)
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool2d(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L2 = tf.nn.dropout(L2,keep_prob=keep_prob)

# L3 ImgIn shape = (?, 7, 7, 64)
W3 = tf.Variable(tf.random.normal([3,3,64,128],stddev=0.01)) #3x3의 필터, 들어온개수, 출력개수(필터개수)

# Conv -> (?,7,7,128)
# Pool -> (?,4,4,128)
# Reshape -> (?,4*4*128) # Flatten them for FC
L3 = tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool2d(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L3 = tf.nn.dropout(L3,keep_prob=keep_prob)
# 여기까지 레이어 통과한 걸 Fully-Connected-Layer에 넣을 것이다.
# 그러므로 reshape 해준다.
L3 = tf.reshape(L3,[-1,128*4*4])

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4",shape=[128*4*4,625],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([625]))
L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
L4 = tf.nn.dropout(L4,keep_prob=keep_prob)

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5",shape=[625,10],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random.normal([10]))
hypothesis = tf.matmul(L4,W5)+b5
learning_rate = 0.001
training_epochs = 15
batch_size = 100

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Learning Started. It takes sometime.")

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs,Y:batch_ys,keep_prob:0.7}
        c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
        avg_cost += c/total_batch
    print("hypothesis:",sess.run(hypothesis,feed_dict=feed_dict))
    print("hypothesis.shape:",sess.run(hypothesis,feed_dict=feed_dict).shape)
    print("Epoch:",epoch,"cost:",avg_cost)
print("Learning Finished!")

print("Testing...")
correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
print("Deep CNN Accuracy:",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels,keep_prob:1}))
# 99.38!


