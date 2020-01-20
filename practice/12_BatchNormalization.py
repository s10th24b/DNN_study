import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Batch Normalization Layer이란 무엇인가?
# activation layer 전에 출력을 normalize하는 레이어이다.

왜 하는가?
각 레이어의 입력 분포는 변한다. 왜냐하면 gradient descent에 의해 W를 업데이트 하면서 이전 레이어의 W가 바뀌기 때문. 
이것은 covariance shift라고 무르며 네트워크의 학습을 어렵게 만든다.
예를 들면 만약 activation layer가 relu 레이어이고 activation layer의 입력이 0보다 작도록 shift되었다면, 어떤 W도 activated되지 않을 것이다!
우리가 배치 정규화가 필요하지 않다면, 파라미터들은 정규화 단계의 offset 처럼 업데이트될거라는 것.

결론: 항상 batch normalization을 해라!

print(mnist.train.images.shape)

# 미완성...
