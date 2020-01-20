import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

# x_train = tf.placeholder(tf.float32)
# y_train = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]),name='weight') # 처음엔 랜덤한 값
b = tf.Variable(tf.random_normal([1]),name='bias')   # 처음엔 랜덤한 값
# Variable은 기존의 변수와 좀 다름. 텐서플로우가 사용하는 Variable이다. 우리가 아니라.
# 텐서플로우가 자체적으로 변화시키는 변수. = trainable 학습하는 과정에서 알아서 변화시킨다.

hypothesis = x_train * W + b
    

cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_mean은 t = [1,2,3,4] 일때  tf.reduce_mean(t) ==> 2.5 로만들어 주는 것. 평균을 내주는 것.

# Cost function = Loss function
# H(x) = Wx + b
# H(x) - y 는 X. 마이너스도 될 수 있기에.
# 그러므로, ( H(x) - y )^2 
# [ H(x1) - y1) ]^2 + [ H(x2) - y2 ]^2 + ... + [ H(xm) - ym ]^2 을 m으로 나눈 것(즉, 평균)이 바로 cost function

# cost(W,b) = cost function 가장 작은 W와 b를 구하는게 Linear Regression의 학습
# minimize cost(W,b)

# Gradient Descent 경사 하강법
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph
# Variable을 만들어서 실행하기 전에는 반드시 global_variables_initializer를 실행해야 한다.
sess.run(tf.global_variables_initializer())

# Fit the Line
for step in range(2001):
    sess.run(train) # train 노드가 cost, 그리고 node는 또 hypothesis 등등 연결되어있으므로 train만 실행시켜도 돌아간다.
    if step % 20 == 0:
        print('step:',step,'cost:',sess.run(cost),'W:',sess.run(W),"b:",sess.run(b))






