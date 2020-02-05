import tensorflow as tf
# rnn의 강점은 serial
# 지금까지의 소스에서는 sequence length가 다 정해져있었다.
# 실전에서는 정해지지않을 가능성이 높음.
# 가변 시퀀스를 입력으로 받아들임
# 기존에는 padding이란 특별기호 넣어서 햇는데 그렇다해도 weight가 들어잇기에
# 헷갈릴수잇음
# 그래서 탠서플로우에서는 batch마다 시퀀스 정의를 위해 sequence_length를 list로.

x_data = np.array([[[///]]],dtype=np.float32)
hidden_size = 2
cell = tf.contrib.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell,x_data,sequence_length[5,3,4],dtype=tf.float32)
# 위 코드처럼, sequence_length를 list로, 5면 꽉 차지만 3이나 4는 해당 row는
# 0으로 된다. 확실하게 없는데이터는 0으로.
sess.run(tf.global_variables_initializer())
print(outputs.eval())

