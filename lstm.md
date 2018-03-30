# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:23:02 2018

@author: Admin
"""
#用LSTMl来做手写识别数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist=input_data.read_data_sets('/data/mnist',one_hot=True)
#创建批次大小
n_input=28
n_classes=10
batch_size=50
max_time=28
lstm_size=100
#计算总的批次
n_batch=mnist.train.num_examples//batch_size
#定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
#初始化权值
weights=tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
#初始化偏置
biases=tf.Variable(tf.constant(0.1,shape=[n_classes]))

#定义RNN网络
def RNN(X,weights,biases):
    inputs=tf.reshape(X,[-1,max_time,n_input])#转换x的维度
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(lstm_size)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)#final_state[0]代表cell state;final_state[1]代表hidden state
    return results
#计算输出
pre=RNN(x,weights,biases)
#求交叉熵代价函数
cross=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pre))
#使用优化器
train=tf.train.AdamOptimizer(1e-4).minimize(cross)
#把结果存放在列表中
coo=tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
#准确率
accuracy=tf.reduce_mean(tf.cast(coo,tf.float32))

#迭代
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(6):
        for i in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("iter"+str(epoch)+",testing acc "+str(acc))
            
    




