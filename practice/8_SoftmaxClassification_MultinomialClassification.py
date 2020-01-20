# Multinomial Classification = 여러개의 클래스가 있을때 그걸 예측함.

# Logistic Regression은, 클래스 2개를 나누는 기준선 하나를 긋는것과 같다. = Binary Classification
# 근데 만약 클래스가 2개보다 많다면?
# -> 클래스 만큼의 binary classification 기준선을 그으면 가능하다.
# 예를 들면, A,B,c가 있을때, A or not, B or not, C or not 3개의 기준선.

#   [ W1 W2 W3 ] * [ X1 X2 X3 ]T = W1X1 + W2X2 + W3X3
# ->
#   [ Wa1 Wa2 Wa3 ]   [ X1 ] = [ Wa1X1 + Wa2X2 + Wa3X3 ]    [ Ya^hat ]
#   | Wb1 Wb2 Wb3 | * | X2 | = | Wb1X1 + Wb2X2 + Wb3X3 |  = | Yb^hat |
#   [ Wc1 Wc2 Wc3 ]   [ X3 ] = [ Wc1X1 + Wc2X2 + Wc3X3 ]    [ Yc^hat ]
# 3개의 독립된 classification 알고리즘을 구현해도 되지만 이렇게 하나의 벡터로 처리하면 한번에 계산이 가능하다.
# X --(W)--> □ -> [ vector ]

# a b c 각각의 클래스에 속할 확률(즉, y^hat)을 더하면 모두 1이 되도록. 이것이 softmax
# 1. 0~1의 값이고, 2. 모두 합하면 1이다.
# 그 중에서 가장 큰 확률을 가진 클래스만 보고싶다면(tf.argmax) one-hot encoding

# 여기까지 Hypothesis였음. 이제 cost function
# Cost function = D(S,L) = - sigma ( Li * log(Si) )
# 여기서 S는 S(softmax)(y) (= 0.7 or 0.2 or 0.1), L(Label)은 Y (= 1.0 or 0.0)
# D(S,L)은 이 2개 사이의 값의 차이를 구하게 된다. 위의 시그마 식을 CROSS-ENTROPY라고 함.

#Cross-entropy cost function
# - sigma ( Li * log(Si) ) = - [ sigma ( Li * log( y^hat i) ) ]  -> 쉽게 말하면, cross-entropy는 Ylog(y^hat). sigma 제외한 부분.
# => sigma [ Li * (-log(y^hat i)) ] # 여기서 -log(y)는 우리가 봤었다. logistic regression에서. 0~1의 값에만 관심있음.

# 만약, Y=L=[0 1]T 라면, 정답은 B이다.
# 여기서 2가지 예측을 할 수 있는데, Y^hat = [0 1]T or [1 0]T
# cost function이라는 건, 정답이면 값이 작고, 틀리면 값이 커져서 우리를 혼내줌.
# 위 벡터를 직접 수식에 넣어보자. [0 1]T라면,
# [0 1]T * -log[0 1]T = [0 1]T * [inf. 0]T (-log(y) 그래프 참고) 엘리먼트 곱을 하면? ◎
# = [0 0]T 을 하고 마지막에 엘리먼트들을 모두 합하는 시그마가 남아있다.
# = 0. 이것이 바로 예측이 맞았을 때다.

# 예측이 틀렸을 때는?
# [0 1]T * -log[1 0]T = [0 1]T * [0 inf.]T (-log(y) 그래프 참고)
# = [0 inf.]T을 하고 마지막에 엘리먼트들을 모두 합하는 시그마 계산.
# = inf. 이것이 바로 예측이 틀렸을 때.

# Logistic cost VS cross-entropy
# Logistic Classification에서 cost function, 즉 Logistic cost는
# C( H(x),y ) = ylog(H(x)) - (1-y)log(1 - H(x)) 였다
# 이것과 오늘 배운 cross-entropy의 D(S,L).
# H(x)가 S, 즉 예측값이고 y가 L(라벨)이다. 사실상 같은 식.

# Cost function: Loss = (1/N) sigma ( D (S(WXi + b), Li ) 여기서 training set에 있는 건, X와 L
# 마지막으로 cost를 구했으면 이 cost를 최소화 시키는 값. W 벡터를 찾아내야한다.
# 이 알고리즘은 저번과 마찬가지로 Gradient Descent 알고리즘 사용.
# 어디서 시작하든 경사면을 따라 가면 cost의 최솟값을 찾을 수 있음.
# 경사면은 이 함수를 미분하는 것. 그러나 오늘 배운 cost, 즉 Loss function은 미분하기 복잡함. 고로 직접 다루지는 않음.
# 기울기를 타고 얼마나 내려갈지를 결정하는 것은 alpha, 즉 learning_rate     
# STEP = -α*ΔL(w1,w2)
