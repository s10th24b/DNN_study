import tensorflow as tf
import numpy as np

# 지금까지 한건 다 복잡했다. 지금까지는 블록 조각들을 배웠고 이젠 블록을 만드는 방법.
# 이제부터는 파이썬의 클래스로 깔끔하게 정리할 것이다.

class Model:
    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            #input placeholder
            self.X = tf.placeholder(tf.float32,[None,784])
            # img 28x28x1
            X_img = tf.reshape(self.X,[-1,28,28,1])
            self.Y = tf.placeholder([None,10])

            #L1 ImgIn shape=(?,28,28,1)
            W1 = tf.Variable(tf.random.normal([3,3,1,32],stddev=0.01))
    def predict(self,x_test,keep_prob=1.0):
        return self.sess.run(self.logits,feed_dict={self.X:x_test,self.keep_prob:keep_prob})


    def get_accuracy(self,x_test,y_test,keep_prob=1.0):
        return self.sess.run(self.accuracy,feed_dict={self.X:x_test,self.Y:y_test,keep_prob:self.keep_prob})

    def train(self,x_data,y_data,keep_prob=0.7):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X:x_data,self.Y:y_data,self.keep_prob:keep_prob})

sess = tf.Session()
m1 = Model(sess,"m1")

sess.run(tf.global_variables_initializer())
print("Learning Started!")

training_epochs = 15
batch_size = 100
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        c,_ = m1.train(batch_xs,batch_ys) #여기서 도움함수를 바로 호출해버림.
        # 복잡한 시스템 만들때 클래스 활용하면 아주 편리.
        avg_cost+=c/total_batch

# tf.conv2d
# tf.dense
# tf.max_pooling2d
