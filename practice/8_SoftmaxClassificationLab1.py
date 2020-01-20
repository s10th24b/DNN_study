import tensorflow as tf
import numpy as np

# softmax function이라는 건 여러개의 클래스를 예측할때 아주 유용하다.
# binary classification은 0이냐 1이냐만 예측하는데 실생활에서는 2개보다 n개 예측하는 경우가 많다.


# 2.0   -> S(yi) = e^(yi)/Σe^(yi) -> 0.7
# 1.0   -> S(yi) = e^(yi)/Σe^(yi) -> 0.2
# 0.1   -> S(yi) = e^(yi)/Σe^(yi) -> 0.1
# y=XW=Scores(=logit) -> Softmax          -> probabilities

# 그렇다면 이걸 텐서플로우로 어떻게 구현할 것인가? -> 간단하다
# hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
# tf.matmul(X,W)+b   -> tf.nn.softmax  ->  probabilities

# cost function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]] # one-hot encoding

X = tf.placeholder("float",[None,4])
Y = tf.placeholder("float",[None,3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(Logits) / reduce_sum(exp(Logits),dim)
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

#Cross-entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if step % 100 == 0:
            print("step:",step,"cost:",sess.run(cost,feed_dict={X:x_data,Y:y_data}))
    predicted =  sess.run(hypothesis,feed_dict={X:x_data,Y:y_data})
    print("predicted:",predicted)
    predicted = sess.run(tf.argmax(predicted,1)) #2차원 배열의 경우 argmax는 그 이하인 0과1이 가능. 0은 열단위, 1은 행단위로 최대값찾음
    print("one-hot encoded predicted:",predicted)
    new_predicted = np.ndarray([8,3])
    print("new_predicted:",new_predicted)
    for i in range(len(predicted)):
        temp = predicted[i]
        print("predicted[i]:",predicted[i])
        new_predicted[i] = np.zeros(3)
        new_predicted[i][temp]+=1
    print("reshaped one-hot encoded predicted:",new_predicted)
    print("acc:",sess.run(tf.reduce_mean(tf.cast(tf.equal(new_predicted,y_data),dtype=tf.float32))))
