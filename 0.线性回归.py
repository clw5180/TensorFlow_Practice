import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1、加载数据，设置超参数及数据预处理
learning_rate = 0.01
training_epochs = 2000
display_step = 50

train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                        7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]
print('train_X:', train_X)
print('train_Y:', train_Y)

# 2、搭建网络
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")
pred = tf.add(tf.multiply(X, W), b)

# 3、计算损失并设置梯度下降的方式
cost = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2 * n_samples)  # 设置cost为均方差
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 4、开始训练
# 使用global_variables_initializer()函数，而不是手动初始化每个变量。
# 请注意，它实际上没有立即执行初始化，而是在图谱中创建一个当程序运行时所有变量都会初始化的节点：
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # 首先初始化所有Variables，即权重和偏置

    # 灌入训练数据
    for epoch in range(training_epochs):
        for(x, y) in zip(train_X, train_Y):  # 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，返回由这些元组组成的对象，节约内存
                                             # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同
            sess.run(optimizer, feed_dict={X:x, Y:y})

        #打印出每次迭代的log
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print('Epoch:', '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),
                "w=", sess.run(W), "b=", sess.run(b))

    print('Optimizion Finished!')
    train_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
    print('Training cost=', train_cost, "w=", sess.run(W), "b=", sess.run(b))

    # 6、作图
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()