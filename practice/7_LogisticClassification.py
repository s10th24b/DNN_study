import tensorflow as tf

# sigmoid function: g(z) = [ 1/(1+e ^(-z)) ] 
# : 항상 1과 0 사이에만 있기에 logistic function이라고도 부름.
# 0 < g(z) < 1 ( z = y hat = H(X) = WX )
# -> 그러므로, 0 or 1 Classification 가능하다!

# Logistic Hypothesis
# H(X) = 1/(1 + e^(-W^(T) * X))
# 이 cost function을 그려보면 울퉁 불퉁함... local minimum에 converge할 위험이 있음 (convex function이 아님)

#New cost function for logistic
# cost(W) = (1/m) sigma [ C (H(x),y) ]
#C(H(x),y) = -log(H(x))     : y = 1
#            -log(1-H(x))   : y = 0
# 왜 log함수가 나오냐. sigmoid 함수는 자연상수 e를 쓰기에, 그와 상극인 로그함수를 쓴다.
# 그리고 실제로 로그함수의 형태를 보면 우리와 잘맞는다.
# cost함수의 뜻이, 예측한 값과 실제 값이 같거나 차이가 적으면  cost는 작아지고 
# 차이가 크면 cost도 커진다.

#만약 y=1 이고, H(x)도 1이다. 즉, 맞았다. 그러면 cost는? -log(1) 이니, 0.
#만약 y=1 이고, H(x)는 0이다. 즉, 틀렸다. 그러면 cost는? -log(0) 이니, 양의 무한대로 커진다 (log(0)이 음의무한대)

#만약 y=0 이고, H(x)도 0이다. 즉, 맞았다. 그러면 cost는? -log(1) 이니, 0.
#만약 y=0 이고, H(x)는 1이다. 즉, 틀렸다. 그러면 cost는? -log(0) 이니, 양의 무한대로 커진다 (log(0)이 음의 무한대)

# -log(x)와 -log(1-x) 이 두개 그래프를 붙이면 convex function처럼 밥그릇 모양이 되어 안전해진다! 경사하강법 적용 가능해짐

# if문을 쓰지 않도록 cost함수를 정리한 것은 다음과 같다.
# C(H(x),y) = -ylog(H(x)) - (1-y)log(1-H(x))
# 복잡해 보이지만 간단하다.
# y = 1일때, -log(H(x)), y = 0일때, -log(1-H(x))

#그 다음 단계. cost가 주어졌으면 이 cost를 minimize. 우리가 좋아하는 경사하강법
# 경사하강법을 할때는 함수의 기울기를 구하기 위해 미분을 하게 되는데, 복잡하므로 직접 할필요 없음.

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
# 0 : fail, 1 : pass

#placeholders for a tensor that will be always fed.
# 이제부턴 shape에 주의하도록.
X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

#Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X,W) + b))
hypothesis = tf.sigmoid( tf.matmul(X,W)+b ) # = H(X)

#cost/loss function
#cost(W) = -(1/m) * sigma [ y*log(H(x)) + (1-y)(log(1-H(x))) ]
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost) #미분 직접 안해도 됨.

#Accuracy Computation
#True if hypothesis >0/5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # True나 False가 나오는데 이걸 float32로 Cast -> 0.0 or 1.0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

#Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([cost,train],feed_dict={X:x_data,Y:y_data})
        if step % 200 == 0:
            print("step:",step,"cost:",cost_val)

    #Accuracy report
    h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
    print("\nHypothesis:",h,"\nCorrect:",c,"\nAccuracy:",a)
