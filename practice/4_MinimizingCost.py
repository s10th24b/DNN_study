import tensorflow as tf
import matplotlib.pyplot as plt

# H(x) = Wx + b
# [ H(x1) - y1) ]^2 + [ H(x2) - y2 ]^2 + ... + [ H(xm) - ym ]^2 을 m으로 나눈 것(즉, 평균)이 바로 cost function
# cost(W,b) = cost function 가장 작은 W와 b를 구하는게 Linear Regression의 학습
# minimize cost(W,b)


# Simplify
# H(x) = Wx
# [ W x1 - y1) ]^2 + [ W x2 - y2 ]^2 + ... + [ W xm - ym ]^2 을 m으로 나눈 것(즉, 평균)이 바로 cost function

# Gradient Descent Algorithm = Minimize cost function
# Do until you converge to a local minimum
# Where you start can determine which minimum you end up

# W에 대해 미분하기 위해 m이 아닌 2m으로 나눠보자.
# cost(W) = (1/2m) *  ( [ W x1 - y1) ]^2 + [ W x2 - y2 ]^2 + ... + [ W xm - ym ]^2 )
# 미분한 최종값이 나온다! 고딩수준 수학
# W := W - alpha(1/m) sigma[i=1 to m] (W*X - Y) * X
# 여기서 alpha (learning_rate)하고 그 뒷부분이 다 델타(세모모양), 즉 기울기이다.
# minimum에서 오른쪽 부분은 기울기가 +이다. 그러므로, W := W-delta 시, delta가 +이므로, W는 - 방향으로 움직인다.
# 그 반대로, 왼쪽 부분은 기울기가 - 이므로, W := W-delta 시, delta가 -이므로, W는 +방향으로 움직인다.

# Convex function. Gradient Descent Algorithm이 항상 같은 답을 찾는 함수다. 다행히 위의 우리의 식은 convex function임.
# Linear Regression이 Convex Function의 형태를 띄고있는 걸 반드시 확인 후, 맞으면 안전하게 Gradient Descent Algorithm 적용가능
# 위험한 건, convex function이 아닌 함수도 있다는 거다.


X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)
# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Launch the graph in a session
sess = tf.Session()
# Initialzes global variables in the graph
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
for i in range(-30,50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost,W], feed_dict={W: feed_W})
    print("curr_cost:",curr_cost)
    print("curr_W:",curr_W)
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val,cost_val)
plt.show()

