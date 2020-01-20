import tensorflow as tf
import numpy as np


# One logistic regression unit cannot separate XOR
# But, Multiple logistic regression CAN
# But, No one on earth had found a viable way to train
# XOR using NN


# x1  x2  XOR
# 0   0   0-
# 0   1   1+
# 1   0   1+
# 1   1   0-

# Linear하게는 구분 가능한 선을 긋지 못한다.
# 이제 3개의 NN을 이용하여 풀 것이다.
# 1개는 2개 입력, WX+b로 계산하고 그거에 sigmoid를 먹인 Net

# Net 1 x1,x2(W=[5 5],b=-8) -> y1
# Net 2 x1,x2(W=[-7 -7],b=-3) -> y2
# Net 3 y1,y2(W=[-11 -11],b=-6) -> y^hat

# x1  x2  y1  y2  y^hat   XOR
# 0   0   0   1   0       0   good
# 0   1   0   0   1       1   good
# 1   0   0   0   1       1   good
# 1   1   1   0   0       0   good

# Forward Propagation

# 그렇다면 저 값 말고 다른 W,b의 조합이 존재할까?
# Recap Multinomial Classification. 3개의 벡터를 하나로 합쳤었지.
# 이것도 마찬가지. W[5 5] b=-8와 W[-7 -7] b=3을 합쳐서 W1[[5,-7],[5,-7]], b1=[-8,3]

# 일단 Net 1,2만 생각해보자. k라는 출력을, k(x) = sigmoid(XW1+b1)
# 그리고나서 Y^hat = H(x) = sigmoid( k(x)W2+b2)
# K = tf.sigmoid(tf.matmul(X,W1)+b1)
# hypothesis = tf.sigmoid(tf.matmul(K,W2)+b2)

# -> 그럼 학습 데이터에서 W1 W2 b1 b2를 어떻게 학습할 수 있을까?
# 일단 hypothesis를 얻었으니 cost함수를 정의해야한다. 만약 이게 밥그릇처럼 생긴 convex function이라면,
# cost함수를 미분하는 Gradient Descent기법으로 기울기를 구해 cost를 최소화시킬 수 있다.
# 이 미분을 계산해야한다
# 각각의 x들이 y끼치는 영향, 즉 미분값을 알아야 각각의 W(weight)를 계산해야하는데 실제로 번거롭고 어려움.
# 그래서 민스키 교수가 아무도 못한다고 한거임.
# 나중에 폴과 힌튼 교수에 의해 해결된다. 바로 Backpropagation.
# 예측값과 실제값을 비교해서 오류, 즉 코스트를 뒤에서 앞으로 거꾸로 뭘 조정해야하는지 계산하겠다는 것.
# Back propagation (chain rule)
# ex) f = wx+b, g = wx, f = g+b
# 여기서 w,x,b가 f에 미치는 영향을 구하고 싶다면, f에 대한 각 요소의 편미분값이 될 것이다.
# 여기서 chain rule. f(g(x))를 x에 대해 편미분한 건, f를 g에대해 편미분한 값과 g를 x에 대해 편미분한 값을 곱한것과 같다.

# 2가지 방법으로 나눠서 진행. 
# 1. forward prop. 학습데이터에서 가져온 값으로 그래프에 값을 입력시킴.
# 2. backward prop. 각 요소가 f에 미치는 실제 미분값을 계산. (편미분을 이용하여, chain rule 적용)
# 그렇게되면 w에 대한 f의 (편)미분값이 5가 된다.이는 w가 1만큼 바뀔때 f에는 5만큼 영향을 준다는 건데, 숫자를 보니 맞는 것 같다
# 이렇게 f에 대한 조정이 가능한 것이다.

# sigmoid 미분. g(z) = 1/( 1+e^(-2) ). 알고싶은 건 g에 z가 어떻게 영향을 끼치는지.
# 이걸 미분하려고 애쓸 필요 없다. 그래프로 그려보면. 
# z -> *(-1) -> exp -> +1  -(x)->  1/x -> g
# 이걸 backpropagation 하면서 미분하면 된다. 그렇게 미분한 값들을 뒤로 곱해나가면 된다 아무리 복잡해도 기계적으로 가능하다.
# 이 그래프는 텐서보드에서 구현해놨다. 우리가 할 필요없음.
