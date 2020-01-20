import tensorflow as tf
import matplotlib.pyplot as plt


x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]),name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# cost function을 미분하면?
# ->  W := W - alpha(1/m) sigma[i=1 to m] (W*X - Y) * X

# Minimize: Gradient Descent using derivative: W -= Learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y) * X)
descent = W - learning_rate * gradient
# descent가 새로운 W값.
update = W.assign(descent)
# assign을 통해 새로운 값 정의 텐서플로우에서는 바로 = 로 assign 못함. 그래서 함수를 통해서.
#update를 실행시키면 이 일련의 동작들이 일어나게 된다.

#Minimize: Gradient Descent Magic
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train = optimizer.minimize(cost)
 # 위 두개를 쓰면 미분할 필요없이 매직을 이용해 편하게 계산 가능.

# Launch the graph in a session
sess = tf.Session()
# Initialzes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(21):
    #update 라는 걸 실행.
    print("update:",sess.run(update,feed_dict = {X:x_data,Y:y_data}))

    print("step:",step,"cost:",sess.run(cost,feed_dict={X:x_data,Y:y_data}),"W:",sess.run(W))
