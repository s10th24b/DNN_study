import tensorflow as tf
import numpy as np

t = np.array([0.,1.,2.,3.,4.,5.,6.,])

# np.pprint(t)
print(t.ndim) #rank
print(t.shape) #shape
print(t[0],t[1],t[-2])
print(t[2:5],t[4:-1])
print(t[:2],t[3:])

t = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
print(t.ndim) # rank
print(t.shape) # shape

# shape, rank, axis(축)
# rank는 array의 처음 시작하는 대괄호가 몇개인지 세면 된다.
# 한편, rank가 3이라면, shape는 (?,?,?) 이렇게 생겼을텐데, 일단 맨 마지막거는 가장 안쪽 bracket의 엘리먼트 개수.
# 이렇게 뒤에서부터 세어나가면 된다.
t = tf.constant([1,2,3,4])
sess = tf.Session()
with sess.as_default():
    assert tf.get_default_session() is sess
    print(tf.shape(t).eval())
    # -> [4]
    t = tf.constant([[1,2],
                 [3,4]])
    print(tf.shape(t).eval())
    # -> [2, 2]

    rank = 4
    if rank == 4:
        shape = (1,2,3,4)
    # 또 Axis 축이라는 개념. 만약 rank가 4라면, 가장 안쪽에 있는 axis는 3. 가장 바깥쪽의 axis는0

    # Matmul VS multiply

    matrix1 = tf.constant([[1.,2.],[3.,4.]]) #rank = 2,shape: [2,2]
    matrix2 = tf.constant([[1.],[2.]]) #rank = 2,shape=[2,1]
    print("Matrix 1 shape:",matrix1.shape)
    print("Matrix 2 shape:",matrix2.shape)
    print("matmul:",tf.matmul(matrix1,matrix2).eval(())) # shape of matrix1 * matrix2 = [2,2] * [2,1] = [2,1]
    print("muliply:",(matrix1*matrix2).eval(())) # Broadcasting. WARNING!!
    # Broadcasting은 잘 사용하면 유용하나 실수로 쓰면 치명적인 결과.
    print("add:",(matrix1+matrix2).eval(())) # Broadcasting. WARNING!!
    # 그런데, shape이 다르더라도, 연산을 가능하게 해주는게 바로 이 브로드캐스팅
    matrix1 = tf.constant([[1.,2.]]) #rank = 2, shape: [1,2]
    matrix2 = tf.constant(3.) #rank = 0, shape:[]
    print("add:",(matrix1+matrix2).eval()) #[4,5]
    matrix1 = tf.constant([[1.,2.]]) #rank = 2,shape: [1,2]
    matrix2 = tf.constant([3.,4.]) #rank = 1,shape=[2]
    print("add:",(matrix1+matrix2).eval()) #[4,5]
    matrix1 = tf.constant([[1.,2.]]) #rank = 2,shape: [1,2]
    matrix2 = tf.constant([[3.],[4.]]) #rank = 1,shape=[2]
    print("add:",(matrix1+matrix2).eval()) #[4,5]

    #reduce_mean
    print("reduce_mean:",tf.reduce_mean([1,2]).eval()) #평균을 구하는데, 줄여서 구한다는 뜻. 여러 값을 하나로 줄여준다는 의미.
    #결과: 1. 원래는 1.5여야 하는데, 안쪽 엘리먼트가 float여야 한다!
    print("reduce_mean:",tf.reduce_mean([[1.,2.],[3.,4.]],axis=0).eval()) #axis = 0 열단위
    print("reduce_mean:",tf.reduce_mean([[1.,2.],[3.,4.]],axis=1).eval()) #axis = 1 행단위
    # rank가 2이므로 axis는 과 1이 있을텐데, 0은, 열단위로 보게된다 [2, 3]이 되고 1은 행단위의 [1.5 3.5]가 된다.
    print("reduce_mean:",tf.reduce_mean([[1.,2.],[3.,4.]],axis=-1).eval()) #axis = -1 가장 안쪽에 있는 axis. 여기서는 1과 같음
    print("reduce_mean:",tf.reduce_mean([[1.,2.],[3.,4.]]).eval()) #axis 를 생략하면? 모두를 평균화해라.

    x = [[1.,2.],
         [3.,4.]]
    print("reduce_sum:",tf.reduce_sum(x).eval()) #axis 를 생략하면? 모두를 더해라.
    print("reduce_sum:",tf.reduce_sum(x,axis=0).eval()) #axis 0? 열단위로 더해라
    print("reduce_sum:",tf.reduce_sum(x,axis=1).eval()) #axis 1? 행단위 평균화해라.
    print("reduce_sum:",tf.reduce_sum(x,axis=-1).eval()) #axis -1? 가장 안쪽을 평균화해라.
    print("reduce_mean(reduce_sum):",tf.reduce_mean(tf.reduce_sum(x,axis=-1)).eval()) #가장 많이 쓰는 형식. 제일 안쪽을 다 더해서 평균내는.
    # 왜 많이 쓰는고 하니, 가장 안쪽거가 실제 atomic 데이터일 확률이 높기 때문에(뇌피셜), 이렇게 편리하게 쓰는듯.
    y=[[[1.],[2.],[3.],[4.],[5.]]]
    print("reduce_sum:",tf.reduce_sum(y,axis=-1).eval()) #axis -1? 가장 안쪽을 평균화해라.
    # y 그대로 나온다


    #argmax
    x = [[0.,1.,2.],
         [2.,1.,0.]]
    print("argmax:",tf.argmax(x,axis=0).eval())
    print("argmax:",tf.argmax(x,axis=1).eval())
    print("argmax:",tf.argmax(x,axis=-1).eval())

    #reshape ** 복잡
    t = np.array([[[0.,1.,2.],
                 [3.,4.,5.]],
                 [[6.,7.,8.],
                 [9.,10.,11.]]])
    print("t.shape:",t.shape) #2,2,3. 가장 안쪽에 있는 값이 3. 보통 건드리지 않음. 그래서 reshape 할때도 마지막은 가만히 3.
    # print("t:",t)
    # 중간에 저렇게 나누지 말고 쭉 늘어뜨리고 싶다.
    t = tf.reshape(t,shape=[-1,3]).eval() # 마지막은 똑같이 3. 앞에거는 알아서 하라고 -1
    print("t:",t)
    t = tf.reshape(t,shape=[-1,1,3]).eval()
    print("t:",t)

    # Reshape ( squeeze, expand )
    print("squeeze:",tf.squeeze([[0],[1],[2]]).eval()) #squeeze는 쫙 펴준다.
    print("expand_dims:",tf.expand_dims([0,1,2],1).eval()) #여기에 차원을 추가하고 싶으면 뒤에 숫자넣음.

    # One hot
    t = tf.one_hot([[0],[1],[2],[0]],depth=3)
    print("one_hot:",t.eval()) #총 클래스가 몇개인지가 바로 depth. 010 이렇게 된 array를 아래로 늘어뜨린걸 상상.
    # one_hot에 넣을때만 해도 rank가 2였는데, one_hot을 거치니 rank가 3으로 늘어났다. 자동으로 하나 더 expand를 하게된다.
    t = tf.reshape(t,shape=[-1,3]).eval() #그게 싫으면 reshape
    print("after tf.reshape t:",t)
    print("after squeeze t:",tf.squeeze(t).eval())

    # Casting
    print("cast:",tf.cast([1.8,2.2,3.3,4.9],tf.int32).eval())
    print("cast:",tf.cast([True,False,1==1,0==1],tf.int32).eval())
    # 주로 Accuracy 검사할때. Y = Y^hat과 같은지 검사.

    # Stack
    x = [1,4]
    y = [2,5]
    z = [3,6]
    # Pack along first dim.
    # Axis에 대한 이해가 중요하다!
    print("stack:",tf.stack([x,y,z]).eval())
    print("stack axis0:",tf.stack([x,y,z],axis=0).eval())
    print("stack axis1:",tf.stack([x,y,z],axis=1).eval())
    print("stack axis-1:",tf.stack([x,y,z],axis=-1).eval())

    # Ones and Zeros like
    x = [[0,1,2],
         [2,1,0]]
    # 숫자로 채우고 싶지만 모양이 똑같은 것으로!
    print("ones_like:",tf.ones_like(x).eval())
    print("zeros_like:",tf.zeros_like(x).eval())

    # Zip
    # 많이 사용하게 된다!
    print("x,y")
    for x,y in zip([1,2,3],[4,5,6]):
        print(x,y)
    print("x,y,z")
    for x,y,z in zip([1,2,3],[4,5,6],[7,8,9]):
        print(x,y,z)
