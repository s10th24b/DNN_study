import tensorflow as tf

X = tf.placeholder([None,784],tf.float32)
X_img = X.reshape([-1,28,28,1])
# Convolution
W1 = tf.Variable(tf.random.normal([3,3,1,32],stddev=0.01))
L1 = tf.nn.conv2d(X_img,W1,strides=[1,1,1,1],padding='SAME')
L1 = tf.nn.relu(L1)
# ->
conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding="SAME", activation=tf.nn.relu)
# input, filters, kernel_size, padding, activation

# Pooling
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# ->
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], padding="SAME", strides=2)

# Dropout
L1 = tf.nn.dropout(L1,keep_prob=self.prob)
# ->
dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

W2 = tf.Variable(tf.random.normal([3,3,32,64],stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
# ->
conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
# inputs, filters, kernel_size, padding, activation

# Pooling
L2 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# ->
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],padding='SAME',strides=2)
# inputs, pool_size, padding, strides

# Dropout
L2 = tf.nn.dropout(L2,keep_prob=self.keep_prob)
# ->
dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)
# inputs, rate, training
# .
# .
# .
# 이제 FC.
flat = tf.reshape(dropout2, [-1,64*3*3])
dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu) #13_DeepCNNLab.py에서, xavier로 W초기화 후, relu적용하는 부분.
# inputs, units, activation
dropout4 = tf.layers.dropout(inputs=dense4,rate=0.5,training=self.training)
