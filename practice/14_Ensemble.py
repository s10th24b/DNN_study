import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 모델 여러개를 조합해서 조화롭게.
# 독립된 모델 여러개를 훈련시키고
# 학습데이터가 들어오면 각각 모델을 예측시키고 그 결과를 어떤방법으로 조합.
# 그 최종결과 내놓으면 좋은 성능을 내놓더라.
# 이걸 만드려면?
# 일단 독립된 모델을 여러개 만들어야 한다.
# class로 하면 편하다.

class Model:
    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)


            #input placeholder
            self.X = tf.placeholder(tf.float32,[None,784])
            # img 28x28x1 , Input Layer
            X_img = tf.reshape(self.X,[-1,28,28,1])
            self.Y = tf.placeholder(tf.float32,[None,10])

            # Conv 1
            conv1 = tf.layers.conv2d(inputs=X_img,filters=32,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            # Pool 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2], padding="SAME", strides=2)
            # Dropout 1
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            # Conv 2
            conv2 = tf.layers.conv2d(inputs=dropout1,filters=64,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            # Pool 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2], padding="SAME", strides=2)
            # Dropout 2
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            # Conv 3
            conv3 = tf.layers.conv2d(inputs=dropout2,filters=128,kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
            # Pool 3
            pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2], padding="SAME", strides=2)
            # Dropout 3
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)
            print("dropout3:",dropout3)
            # (?,4,4,128)
            # print("Enter Anykey to proceed")
            # temp_pause = input()

            # 이제 FC.
            # Dense Layer with ReLu
            flat = tf.reshape(dropout3, [-1,128*4*4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu) #13_DeepCNNLab.py에서, xavier로 W초기화 후, relu적용하는 부분.
            # inputs, units, activation
            dropout4 = tf.layers.dropout(inputs=dense4,rate=0.5,training=self.training)

            # Logits(no activation) layer = L5 Final FC 625 inputs  ->  10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

    def predict(self,x_test,training=False):
        return self.sess.run(self.logits,feed_dict={self.X:x_test,self.training:training})

    def get_accuracy(self,x_test,y_test,training=False):
        return self.sess.run(self.accuracy,feed_dict={self.X:x_test,self.Y:y_test,self.training:training})

    def train(self,x_data,y_data,training=True):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data,self.training:training})

models = []
num_models = 10
num_classes = 10
learning_rate = 0.001
training_epochs = 1
batch_size = 100
sess = tf.Session()
for m in range(num_models):
    models.append(Model(sess,"model"+str(m)))
sess.run(tf.global_variables_initializer())
print("Learning Started!")


# train my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)

        #train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs,batch_ys)
            avg_cost_list[m_idx] +=c/total_batch
            # print(m_idx,"th Model is Trained.")
            # print(m_idx,"th Model's Cost:",avg_cost_list[m_idx])
    print("Epoch:",epoch+1,"Cost:",avg_cost_list)
    lowest_idx = sess.run(tf.argmin(avg_cost_list))
    print("Lowest Cost: Model["+str(lowest_idx)+"]:",avg_cost_list[lowest_idx])
print("Learning Finished!")

# Testing...
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size*num_classes).reshape(test_size,num_classes)
for m_idx, m in enumerate(models):
    print(m_idx,"Accuracy:",m.get_accuracy(mnist.test.images,mnist.test.labels))
    p = m.predict(mnist.test.images)
    print("p:",p)
    predictions+=p
print("predictions:",predictions)
print("predictions.shape:",predictions.shape)
print("mnist.test.labels.shape:",mnist.test.labels.shape)
ensemble_correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(mnist.test.labels,1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction,dtype=tf.float32))
print("Ensemble Acc:",sess.run(ensemble_accuracy))

# 0 Accuracy: 0.9938
# 1 Accuracy: 0.994
# 2 Accuracy: 0.9925
# 3 Accuracy: 0.9935
# 4 Accuracy: 0.9935
# 5 Accuracy: 0.9931
# 6 Accuracy: 0.9942
# 7 Accuracy: 0.9932
# 8 Accuracy: 0.9944
# 9 Accuracy: 0.9943

# Ensemble Acc: 0.9953
# WOWOWOWOWOW!!
