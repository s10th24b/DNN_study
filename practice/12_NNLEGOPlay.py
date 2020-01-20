우리가 지금까지 한건 몇개 단을 쌓아서 차례대로 가는 네트워크 구성했음. Feedforward NN이라고 함.
그런데 이런 형태만 있는게 아니라 굉장히 다양한 구조를 만들 수 있다.

- Fast Forward.
단순한 x-> o -> o -> ... -> y^hat
여기서 1번째 은닉층의 결과를 2단 앞으로 옮겨서 그 결과랑 더하고., 그리고 다른 은닉층 결과를 2단 앞으로... 이렇게 구성할 수 있다. 이걸 fast forward
2015년도에 He가 개발한 ResNet의 네트워크 구조

- Split & Merge
X 가 y^hat으로 가는동안 입력을 나누어서 넣든, 중간에 출력이 여러개로 나뉘다가 합치든...
X를 여러개로 나누어 넣고 나중에는 하나로 합쳐지는 유명한 Convolutional Network가 있지.

- Recurrent Network
근데, 앞으로만 나가지 말고 옆으로도 나가면 안되나?

'The Only Limit is your Imagination'
