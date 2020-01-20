#1. Learning Rate
# cost function을 정의하고 그걸 최소화하는 값을 찾기위해 사용한 경사하강법.
# 이 떄, 알고리즘을 정할때 α(learning_rate)값을 임의로 정했다.
# 이걸 잘 정하는게 중요하다.
# 너무 크면 overshooting. 바깥으로 튕겨나가 발산.
# 너무 작으면 overfitting. 너무 오래걸리고, local minimum에서 멈춘다.
# 우선 0.01 정도로 시작하고 결과에 따라 올리거나 낮춘다.

#2. Data Preprocessing for gradient descent
#  데이터 값에 큰 차이가 있을 경우, 이걸 정규화normalize 시켜줘야한다.
#  그래프로 봤을떄 엄청 sparse한것도 정규화하면 한 범위안에 다 들어가도록 된다.
# x의 값들이 있으면 그 값을 계산한 평균과 분산의 값을 가지고 나누면 된다.
# Standardization -> x_std[:,0] = (X[:,0] - x[:,0].mean()) / X[:,0].std()

#3. Overfitting. 학습데이터에 너무 딱 잘 맞음(기억해버림). 테스트 데이터셋에서는 성능 안좋음.
# 해결책은, training data가 많으면 많을수록 줄일 수 있고, feature의 개수를 줄이는 것도 하나의 방법이다.
# 마지막으로 두가지 방법 외에도 Regularization. 일반화. 일반화시킨다는 건, weight에서 너무 큰 값을 가지지 않도록.
# 선을 구부리지 말고 펴자는 의미. 
# cost 함수의 뒤에 이 term을 추가시켜 준다. λΣW^2 (= regularization strength)
# W의 값, 벡터값(각각 엘리먼트에 제곱을 한 값)이 작아질 수 있도록.
# Loss function = 1/N ΣD(S(WXi+b),Li) + λΣW^2

#-> tensorflow로 구현하려면, l2reg = 0.001 * tf.reduce_sum(tf.square(W))



#모델의 성능 확인방법.
#한번 training set을 먹이고 학습시킨 후, 다시 set을 먹이면서 학습한다 보통. 결국 overfitting된다.
#  training set과 test set을 나눠서 학습한다. test set은 절대 보면 안된다. 숨어있음.
# 완벽하게 학습이 끝났다고 생각했을때 단 한번만 test를 돌려본다
# Training + Testing => Training, Validation + Testing
# Training set 가지고 모델을 학습시킨 다음에 이 Validation set을 가지고 learning rate나, λ와 같은 하이퍼 파라미터들이 어떤것이 좋을까를 튜닝하게 됨.

#Online Learning
# 데이터셋이 굉장히 많으면 다 넣어서 학습하기 힘들다. 그래서 온라인 러닝이라는 형태의 학습.
# 데이터 100만개를 10만개씩 잘라서 학습시킴. 우리 모델이 해야할일은 첫번째 학습된 결과가 남아있어야함.
# 2번째 학습시키면 추가되어 새로운 학습. 이게 굉장히 좋은 모델. 새로운 데이터 들어와도 이전 기억 잊지않고 쌓임.
