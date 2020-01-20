# 드랍아웃과 앙상블을 하는 이유는 Overfitting을 피하기 위함이다
# 내가 overfitting인지 아닌지 어떻게 알지? train에서는 높지만 test에선 낮다.
# 네트워크를 깊게 할수록 overfitting될 가능성이 높아진다
# 왜냐면 굉장히 많은 변수가 사용되고 hyperplane이 꼬불꼬불해져서 ..
# solution: 1. training data를 많게하고,2. 피쳐 개수를 줄이고, 3. 일반화(Regularization)
# 일반화: Weight에서 너무 큰값을 갖지 말자!
# 이걸 수학적으로 표현한게 l2reg = 0.001*tf.reduce_sum(tf.square(W)) (이게 regularization strength)
# -> cost + λ*ΣW^2 

# DroupOut: 그만둬버리는거. A Simple way to prevent NN from Overfitting
# 힘들게 엮어놓은 NN을 끊어서 몇개 노드를 죽이자.
# Regularization: DroupOut. randomly set some neurons to zero in the forward pass
# 이게 진짜 되긴돼? Forces the network to have a redundant representation
# 몇개 노드들을 쉬게하고 나머지 노드들이 맞추게 한다. (각 노드들은 전문가임)
# 그런다음에 마지막에 총동원해서 예측. 이것이 드랍아웃의 아이디어.
# 이걸 Tensorflow에서 구현할때는 한 단만 더 넣어주면 된다.
# dropout_rate = tf.placeholder("float")# dropout_rate몇프로 드랍아웃시킬지를 정해야함. 보통 0.5
# _L1 = tf.nn.relu(tf.add(tf.matmul(X,W)+b))
# L1 = tf.nn.dropout(_L1,dropout_rate) 
# 그런데 주의해야할 점이 이것은 학습시키는 동안에만 dropout 시켜야한다. 실제론? 전문가 모두를 불러와야함. dropout_rate = 1

# 앙상블 Ensemble
# 내가 독립적으로 10단 NN을 만든다. 똑같은 형태의 NN. 만들고 학습을 시킨다. 
# 초기값이 조금씩 다르니까 결과도 조금씩 다를거다. 마지막에 다 합침.
# 실제로 해보면 2%~4%정도 향상됨.

