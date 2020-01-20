import tensorflow as tf
import numpy as np

# print를 계속 하면서 acc, loss 이런걸 다 확인해야했다.
# 하지만 텐서보드를 이용하면 loss acc 이런걸 다 그래프로 확인할 수 있다.

# 간단한 5개의 스텝이 있다.
# 1. TF 그래프에서 어떤 텐서를 로깅하고 싶은지 정해라
    # w2_hist = tf.summary.historm("weights2",W2)
    # cost_summ = tf.summary.scalar("cost",cost)
# 2. 모든 summary를 Merge한다.
    # summary = tf.summary.merge_all()
# 3. writer와 add_graph를 만든다.
# writer = tf.summary.FileWriter('./logs') #파일의 위치를 정하고
# writer.add_graph(sess.graph) #이 세션에 그래프를 넣어준다.
# 4. summary merge와 add_summary를 run시킨다
# s,_ = sess.run([summary,optimizer],feed_dict=feed_dict) #summary도 텐서이기 때문에 session run을 시켜줘야한다.
# writer.add_summary(s,global_step=global_step) #실제로 파일에 기록하는 부분.
# 5. 텐서보드를 실행한다.
# tensorboard --logdir=./logs

# 1.그래프를 보고싶을때 다 펼치면 보기 힘들기에 name_scope로 계층별로 정리 가능.
# with tf.name_scope("layer1") as scope:

# 2,3. 
# summary = tf.summary.merge_all()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# writer = tf.summary.FileWriter('TB_SUMMARY_DIR') #파일의 위치를 정하고
# writer.add_graph(sess.graph) #이 세션에 그래프를 넣어준다.

# 4. summary merge와 add_summary를 run시킨다
# s,_ = sess.run([summary,optimizer],feed_dict=feed_dict) #summary도 텐서이기 때문에 session run을 시켜줘야한다.
# writer.add_summary(s,global_step=global_step) #실제로 파일에 기록하는 부분.
# global_step += 1


# 5. 텐서보드를 실행한다.
# tensorboard --logdir=./logs

#근데 만약 2개의 learning rate를 각각 비교해보고싶다면? Multiple runs
# tensorboard --logdir=./logs/xor_logs
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# ...
# writer = tf.summary.FileWriter('./logs/xor_logs')

# tensorboard --logdir=./logs/xor_logs_r0_01
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# ...
# writer = tf.summary.FileWriter('./logs/xor_logs_r0_01')

# ~> tensorboard --logdir=./logs
# 그럼 127.0.0.1:6006 으로 이동하게된다.

# 원격 tensorboard 팁.
# ssh -L local_port:127.0.0.1:remote_port username@server.com
# local> ssh -L 7007:121.0.0.1:6006 hunkim@server.com
# server> tensorboard --logdir-./logs/xor_logs
# 이렇게 하면 내 컴터에선 7007포트로, 원격으로는 6006 실제 텐서보드 서버로 이동하게 된다.
