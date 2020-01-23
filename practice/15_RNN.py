import tensorflow as tf

# RNN은 가장 재미있고 DNN의 꽃!
# 세상에는 Sequential한 데이터가 굉장히 많음. 음성인식이라든지, 자연어.
# 이전에 했던 것을 반영해야함. 기존의 NN은 하나의 입력이 있으면 이게 바로 출력으로 나타나는 간단한 형태
# 이런 형태의 series한 데이터들을 처리하기가 어려웠음
# 그래서 만든게 바로 이 RNN
# 이전의 결과가 그 다음에 영향을 미칠 수 있어야 한다.
# 현재의 state가 미래의 state에 영향을 미친다.

# Series에 적합한 구조이다.
# State를 계산하고 그 출력이 다음의 입력이 된다.

# ht = fw(ht-1 , xt)
# ht = new state, ht-1 = old state
# fw = some function with parameters W
# xt = input vector at some time step

# 이전 state와 X 이 2개를 가지고 parameter W를 가진 f에 통과시켜 새로운 state를 계산한다.
# 왜 RNN을 표현할때 보통 fold시켜놓느냐? 저 f가 모든 뉴런에 대해서 동일하다.
# Notice: Same function and the same set of parameters(weight) are used at every time step

# Vanilla RNN (state consists of a single 'hidden' vector h):
    # h_t = f_w(h_(t-1), x_t)
    # -> 
        # # h_(t-1), x_t에 대한 각각의 weight를 만들어준다. -> W_hh, W_xh
        # # weight와 각각 h,x를 곱한다음(WX나 마찬가지) 그 결과를 둘이 더한다
        # # 그 다음에 tanh에 넣는다. (tanh는 sigmoid랑 비슷)
        # h_t = tanh(W_hh * (h_t-1 ) + W_xh*x_t)

        # y를 뽑을때는, 마찬가지로 weight을 곱해준다. -> W_hy (WX나 마찬가지)
        # y_t = W_hy * h_t

        # # 이것처럼, h,y는 각각의 weight의 벡터에 따라 그 벡터 형태가 갈린다.

# Character-level language model example 여기서 하고싶은건, h를 넣으면 e처럼 다음 글자를 예측하도록.
# Vocabulary: [h,e,l,o]
# Example training sequence: "hello"
# 우선 각각의 입력을 벡터로 표현해야한다. 여기서 가장 쉬운건 one-hot encoding h=[1,0,0,0]...
# 맨 처음 state인 h_t-1은 0이나 마찬가지. 고로, 결과값 h_t = tanh(W_xh * x_t)이다.
# W_xh는 아직 모르지 당연히. 학습에 의해 주어진다
# 그렇게 계산하고 각각의 output layer에서는 입력이랑 똑같이 4개가 나오고 거기서 가장 높은거로 label정함.
# cost함수로 softmax 써서 쉽게..

# 그밖에 활용용도가 다양함. 
# language model, 음성인식, 기계번역, 대화모델링, QA, 이미지/비디오 자막달기, 이미지/음악/춤 Generation

# 위에서 설명한 Vanilla RNN은 깊이가 깊어지다보면 학습에 어려움이 있음. 이걸 극복하는 모델이 여러개있음.(LSTM)
# 또는 한국의 조 교수님이 만든 GRU모델.
