# Vanishing Gradient 해결 방법
# 1.ReLU 사용
# 2. 조프리 힌튼의 일침처럼, 우린 초기 weight를 멍청하게 초기화했다..

# 모든 초기 weight를 0으로 해볼까? 그럼 gradient가 0으로 사라져버림. 어떻게 하지?
# 일단 절대로 모든 값을 0으로 하면 안된다. 힌튼이 2006년에 논문을 냈다.
# $ Restricted Boatman Machine (RBM)을 사용. 지금은 잘 안쓰지만 많이 나오는 용어니까 한번 봅시다.

# Restriction = No Connection Within A Layer

# Recreate Input. 입력을 재구성한다. 앞으로 값을 넘겨주는걸 forward라고 한다. 어떤건 activate되고 어떤건 안되고..
# Backward 시에는 weight곱해서 거꾸로.. 여기까지는 다 아는내용.
# 그런데 처음의 x값과 뒤에서 거꾸로 보내면서 생성된 x^hat의 값을 비교한다. 이 둘의 차이가 최저가 되도록 weight를 조절.
# x가 앞으로 가는걸 encode, 출력층에서 거꾸로 x입력층으로 가는걸 decode라고 한다. 여기서 오토인코더, 오토디코더 라고도 한다.
# 이게 바로 볼츠만머신.

# 이걸로 weight를 초기화시키는 거다. 어떻게? 레이어가 많잖아? 많지만 나머지 신경쓰지말고 처음 2개 레이어만 본다.
# 인코드 디코드 반복해서, X^hat값이 내가 준 초기X값과 유사하도록 weight를 학습. 이렇게 학습시키고 그다음층으로 이동해서 반복
# 이걸 특별히 Deep Belief Network라고 한다.
# pre-training과정이 필요하다. 처음엔 2개 층만 가지고 RBM을 돌린다. weight들을 학습. 그리고 다른거 신경끄고 그 다음 2개층으로 계속 반복.. 
# (처음에 입력층,1번은닉층 봤다면 그 다음은 1번은닉층과 2번은닉층)
# 이렇게 진행시키면 다시한번 전체를 보면 weight들이 초기화된 값. 이걸 초기화값으로 이용한다. 굉장히 똑똑하고 실제로도 잘된다
# 그다음 실제로 학습을 돌린다. 근데 초기화가 잘돼서 데이터를 많이 안써도 학습이 빨리 됐다. 그래서 이걸 Fine Tuning이라고도 한다.
# 왜냐하면 이미 weight들이 잘 학습되어있고 조금만 튜닝하면 되니.
# 요약: 초기화값 잘주면 된다. RBM을 사용하세요!

# 좋은 소식. 굳이 RBM안써도 된다. 굉장히 간단한 초기값줘도 되더라!

# - Xavier initialization: 2010년 논문에 나옴. 간단함. 입력에 몇개 입력, 몇개 출력인가에 비례해서 초기값 부여하면 된다.
# 입력(fan_in) 출력(fan_out)
# W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)

# - He's initialization: 2015년 발표.  # 이미지넷의 오류를 3%로 이하로 떨어뜨린 ResNet 논문을 발표했음.
# W = np.random.randn(fan_in,fan_out)/np.sqrt(fan_in/2) #깜짝 놀랄정도로 좋은 성능.
# Lab때 볼거다

