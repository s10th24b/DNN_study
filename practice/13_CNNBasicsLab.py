import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# CNN이 특히 이미지에 강함. 아산병원에서 열린 콘테스트에서 CNN을 이용해 CT이미지 결과를 보는 거 만들어서 1등.

# CNN은 2가지로 나눠볼 수 있다. 주어진 이미지 벡터에 필터를 갖다대어 각각 1개 값을 뽑아내고 이미지 벡터가 출력되고,
# 이것을 sampling 과정을 거친다. 이렇게 2개.

# 우선 간단한 Convolution layer 만들어보자. stride = 1
# 3x3x1 image -> 2x2x1 filter W
# 이 과정을 거치면 몇개의 데이터? -> 2x2x1의 데이터가 나올 것.

# 그럼 이제 이미지를 만들어보자.
# sess = tf.InteractiveSession()
sess = tf.Session()
image = np.array([[[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]],dtype=np.float32)
print(image.shape)
plt.imshow(image.reshape(3,3),cmap='Greys')
plt.show()
# (1, 3, 3, 1) 맨 앞의 1은 1개의 이미지를 뜻함.

# 필터 크기는 2,2,1(color, 원 이미지랑 같아야함),1(필터개수) stride=1x1, padding:valid
# -> 계산해보면 [[12],[16],[...],[...]]

weight = tf.constant([[[[1.]],[[1.]]],[[[1.]],[[1.]]]])
print("weight.shape:",weight.shape)
conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding="VALID") #Zero-padding을 하지 않겠다
conv2d_img = conv2d.eval(session=sess)
print("conv2d_img.shape:",conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img,0,3)
for i , one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1),plt.imshow(one_img.reshape(2,2),cmap='gray')
    plt.show()

conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding="SAME") #출력을 원본 이미지와 같도록 패딩하여 뽑아내겠다 = 우측에 zero열을 하나 만든다.
conv2d_img = conv2d.eval(session=sess)
print("conv2d_img.shape:",conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img,0,3)
for i , one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1),plt.imshow(one_img.reshape(3,3),cmap='gray')
    plt.show()


# 만약 필터를 여러개 쓰고 싶다면?
print("만약 필터를 여러개 쓰고 싶다면?")
image = np.array([[[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]],dtype=np.float32)
print(image.shape)
weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],[[[1.,10.,-1.]],[[1.,10.,-1.]]]])
print("weight.shape:",weight.shape)
# (2, 2, 1, 3) 에서, 마지막 3이 필터의 개수를 뜻한다.
conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding="SAME") #Zero-padding을 하지 않겠다
conv2d_img = conv2d.eval(session=sess)
conv2d_img = np.swapaxes(conv2d_img,0,3)
print("conv2d_img.shape:",conv2d_img.shape)
for i , one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1),plt.imshow(one_img.reshape(3,3),cmap='gray')
plt.show()

# 이제 Pooling을 할 것.
print('이제 Pooling을 할 것.')
image = np.array([[[[4],[3]],[[2],[1]]]],dtype=np.float32)
pool = tf.nn.max_pool(image,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')
# ksize: 4개이상의 크기를 갖는 리스트로 입력 데이터의 각 차원의 윈도우 크기
#ksize는 커널사이즈를 말한다. ksize가 [1,2,2,1]이라는 뜻은 2칸씩 이동하면서 출력결과를 1개 만들어 낸다는 것이다.
# 다시말해 4개의 데이터 중 에서 가장 큰 1개를 반환하는 역할을 한다.
# 2,2 가 2x2를 말함.
print("pool.shape:",pool.shape)
print("pool.eval(session=sess):",pool.eval(session=sess))


# 이제 실제 MNIST데이터에 적용.
print('이제 실제 MNIST데이터에 적용.')
img = mnist.train.images[0].reshape(28,28)
plt.imshow(img,cmap='gray')
plt.show()


sess = tf.Session()
img = img.reshape(-1,28,28,1)
W1 = tf.Variable(tf.random.normal([3,3,1,5],stddev=0.01))
sess.run(tf.global_variables_initializer())
print("W1:",W1)
conv2d = tf.nn.conv2d(img,W1,strides=[1,2,2,1],padding='SAME') # SAME이라도 stride가 2면, 28 -> 14로 나온다.
print("conv2d:",conv2d)
conv2d_img = conv2d.eval(session=sess)
conv2d_img = np.swapaxes(conv2d_img,0,3)
print("conv2d_img.shape:",conv2d_img.shape)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1),plt.imshow(one_img.reshape(14,14),cmap='gray')
plt.show()

print('이제 Pooling을 할 것.')
pool = tf.nn.max_pool(conv2d,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# ksize: 4개이상의 크기를 갖는 리스트로 입력 데이터의 각 차원의 윈도우 크기
#ksize는 커널사이즈를 말한다. ksize가 [1,2,2,1]이라는 뜻은 2칸씩 이동하면서 출력결과를 1개 만들어 낸다는 것이다.
# 다시말해 4개의 데이터 중 에서 가장 큰 1개를 반환하는 역할을 한다.
# 2,2 가 2x2를 말함.
print("pool:",pool)
pool_img = pool.eval(session=sess)
pool_img = np.swapaxes(pool_img,0,3)
print("pool_img.shape:",pool_img.shape)
print("pool_img",pool_img)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1),plt.imshow(one_img.reshape(7,7),cmap='gray')
plt.show()
