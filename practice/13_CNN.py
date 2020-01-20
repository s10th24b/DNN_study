import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 고양이 뇌로 실험한 결과, 하나를 보더라도 각각 부분을 처리하는 뉴런들이 다 여러개로 분리되어있더라. 

# 구조는, CONV->RELU->CONV->RELU->POOL -> CONV->RELU->CONV->RELU->POOL -> CONV->RELU->CONV->RELU->POOL ->FC (FC classify output vectors with softmax)

# initial input: 32x32x3 image 여기서 이미지의 일부분만 처리하고 싶다. (=filter) 5x5x3 filter (마지막 3은 항상 같다)
# 필터가 이미지를 5x5로 잘라서 가져옴. 이 필터는 궁극적으로 한 값을 만들어냄. (5x5x3 -> 1) 어떻게?
# = Wx+b를 이용해 한 값으로 만든다! -> y^hat . 여기서 W가 어떤 숫자로 만들어낼 것인지를 결정하는 필터의 값이라고 생각하면 됨.
# ReLU(Wx+b)를 적용해도 됨.
# 이렇게 필터를 옆으로 넘기면서 각각 값들을 가져오는 형태로. 얼마나 많은 숫자를 가져올 수 있나?
# 7x7 이미지에서 3x3 필터라면? stride가 1이라면 5x5 output. stride가 2라면 3x3 output
# 필터가 한번 움직이는 길이를 Stride라고 한다.
# Output Size: ((N-F)/stride)  + 1 하면 된다.

# 이미지가 점점 작아지더라. 작아질수록 정보를 잃어버리므로, zero padding! 테두리에 값이 0인 픽셀을 추가한다.
# 7x7 이미지를 3x3 필터로 stride=1, 1 pixel로 zero padding하면 output은? -> 7x7. (zeropadding하면 9x9가 되니..)
# 이 이야기는, 이미지를 7x7로 넣고 convolution layer를 실행시켜도, 같은 사이즈로 나온다는 것!
# 그래서 패딩을 해서 원래 입력과 출력의 사이즈가 같아지도록 하는걸 일반적으로 사용한다!

# 32x32x3 을 5x5x3 필터로 훑으면 output이 나오는데, 필터를 1개가 아니라 2개를 각각 쓴다면, output도 2개가 될것.
# 만약 6개 필터를 각각 쓰면 output(activation maps)도 6개로, 조금씩 다를 것이다. 결과의 깊이는 (?,?,6)이 될 것. 마지막은 필터의 개수와 항상같음
# -> 패딩을 안한다고 가정하면, (28,28,6)이 될 것이다.

# 근데 필터링을 한번이 아니라 여러개를 거쳐서 할수도 있겠다.
# 32x32x3 image를 CONV(6개의 5x5x3),ReLU를 한번에 적용한다. 그럼 output 깊이가 6이 될 것.
# 이렇게 다르게 바꿔가며 여러번을. 그럼 다음 필터의 마지막은 무조건 6으로 같아야 하고, 그 필터가(10개의 5x5x6)이라면,
# 그 다음 아웃풋의 깊이는 (?,?,10)이 될 것이다. 이런식으로 Convolutional layer가 진행이 된다.
# 그럼 여기서 사용되는 weight variables 개수가 얼마나 될까?
# 5x5x3에서 3개를 썼으니 3개. 그럼 이 값은 어떻게 정해지나? 처음엔 초기화를 한다. initialize하든 랜덤하든. 그 다음에 학습을 하는 것이다.


# 그 다음은 max pooling pooling이라는 것은 간단하게 sampling이라고 보면 된다. 앞에 이미지가 있고 필터처리하는 CL을 만들었다.
# 여기서 한 레이어만 뽑아낸다. 그럼 종이 한장같은 레이어가 나오게 되는데 그걸 작게 resize를 한다. 이게 바로 pooling
# 간단하다. 이걸 반복해 각각 레이어를 다 풀링한다.
# 4x4의 이미지가 있다면, 2x2 pooling size로 stride=2로 Max Pooling하면, 4개의 출력, 2x2의 output이 된다
# 그럼 원본 이미지의 2x2 -> 1x1이 되는건데... 어떤 값을 가져올 것이냐? 4개 값을 평균내거나, 가장 큰값 or 작은 값을 가져오거나...
# 그런데, 가장 많이 사용되는 방법이 MAX POOLING, 즉 가장 큰 값을 가져오는 것이다.
# 이걸 sampling이라 하는 이유는 전체 값중에 1개만 뽑는다는 뜻에서 하는 것이다.

# 여기까지 conv, pool을 다루었고, 중간의 ReLU는 conv에서 나온 벡터를 relu 함수에 입력하면 나오는 것뿐. 거기서 나온걸 pooling.
# 이걸 어떤 순서로 쌓는지는 우리의 몫이다.
# 맨 마지막에 보통 pooling을 하게되는데 마지막 값이 3x3x10이라면, 이걸 X(입력)이라고 본다. 그걸 원하는 깊이의 일반적인 NN (Fully Connected Layer)에 두어
# Classifying을 하는 형태로 구성한다.

# http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
