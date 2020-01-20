import tensorflow as tf

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

train = tf.train.AdadeltaOptimizer
train = tf.train.AdagradOptimizer
train = tf.train.AdagradDAOptimizer
train = tf.train.MomentumOptimizer
train = tf.train.AdamOptimizer
train = tf.train.FtrlOptimizer
train = tf.train.ProximalGradientDescentOptimizer
train = tf.train.ProximalAdagradOptimizer
train = tf.train.RMSPropOptimizer

# http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html

# Adam이 cost가 가장 빨리 줄어들더라. Adam쓰면 무난함.
