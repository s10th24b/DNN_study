import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]] #slicing -1 -> end

#Make sure the shape and data are OK
print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data,len(y_data))

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

# Launch
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val,hy_val, _= sess.run([cost,W,hypothesis,train], feed_dict={X:x_data,Y:y_data})
    if step % 10 ==0:
        print(step,"cost:",cost_val,"\nW:",W_val,"\nprediction:",hy_val)

print("Your score will be",sess.run(hypothesis,feed_dict={X:[[100,70,101]]}))
print("Other scores will be",sess.run(hypothesis,feed_dict={X:[[60,70,110],[90,100,80]]}))

