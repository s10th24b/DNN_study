import tensorflow as tf

# 사람들이 어떻게 응용했는지에 대해 알아보고 우리는 어떻게 쓸지를 생각해본다.
# 가장 처음 CNN을 구상한 사람은 LeCun이 창시한 LeNet-5

# 그 다음은 AlexNet. 굉장히 유명하고 2012년 많은 사람들의 관심을 끔. 이미지넷 2012 1등.
# 처음 인풋은 227x227x3 images  처음 conv 레이어는 96개 conv, stride 4 11x11
# output volum [55x55x96]. parameter: (11*11*3)*96 = 35K 의 파라미터 필요, 학습.
# 두번째 pool레이어는 3x3 filters, stride 2.
# output volumn [27x27x96], parameter: 0!
# 계속 이어준다. CONV, MAX, NORM, CONV,MAX,NORM,CONV,CONV,CONV,MAX -> FC->FC->FC
# 여기서 ReLU를 처음 사용해서 잘 됨. dropout도 사용. 이런모델을 7개 만들고 그 결과를 합침. ensemble했더니 error:18.2% -> 15.4%

# GoogLeNet. Inception Module 이미지넷 2014 1등.

# ResNet 2015 1등. 최강. 3.6% 에러. He가 개발.
# 놀라운 것이 처음 알렉스넷은 8개의 레이어. 근데, 이건 152개의 레이어를 사용.
# 레이어가 깊어지면..? 학습 어려울 거 같은데... 어떻게 극복? fast-forward 개념을 사용해서 해결.

# 34-layer plain(일반레이어)는 쭈욱 차례로 연결. 근데 ResNet에서는 중간에 값을 한칸 뛰어서 더해짐.
# 전체적인 레이어는 깊지만, 실제로 학습입장에서는 그다지 깊지않은 느낌으로 학습.
# 아직까지 이게 왜 학습이 잘되는지 설명은 어렵다.

# 이미지만 처리하는게 아니라 여러가지가 가능하다. Yoon Kim박사가 2014년에 창의적으로 제안한 텍스트를 CNN으로.
# Text Syntax Classify. NLP에도 잘 사용되고 있다.

# 알파고도 ConvNN 사용.
