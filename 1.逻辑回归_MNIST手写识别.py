import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data
# MNIST的训练集包含55000张图片，验证集包含5000张图片，测试集包含10000张图片
#
# 为了简化，每张图片已经被转换成一个1x784的一维数组，表示784个特征


# 0 MNIST数据集下载
# http://yann.lecun.com/exdb/mnist/


# 1、加载数据，设置超参数及数据预处理
# 训练输入是55000*784的Tensor，对应label是55000*10的Tensor，这里对10个种类进行one-hot编码
mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)
print('clw：mnist.train.images.shape = ', mnist.train.images.shape, 'mnist.train.labels.shape = ', mnist.train.labels.shape)
print('clw：mnist.test.images.shape = ', mnist.test.images.shape, 'mnist.test.labels.shape = ',mnist.test.labels.shape)
print('clw：mnist.validation.images.shape = ', mnist.validation.images.shape, 'mnist.validation.labels.shape = ', mnist.validation.labels.shape)

num_examples = mnist.train.images.shape[0]
learning_rate = 0.01
training_epoches = 25
batch_size = 128
display_step = 1

# 数据预处理
# 这里官方已经做好了数据预处理：28x28尺寸的黑白图片，每个像素值已经标准化到0-1区间，0代表白色，1代表黑色，区间内的值代表灰色；
# 由于这里展成了长度为784一维向量；这里丢弃了二位空间的信息，因此效果肯定不如CNN

# 2 搭建网络：定义Graph的输入，并初始化参数
x = tf.placeholder(tf.float32, [None, 784]) # None代表输入的样本个数是不确定的，784代表每个样本都对应一个784维的向量
W = tf.Variable(tf.zeros([784, 10])) # 注意这里可以全部初始化为0，但对于CNN或者比较深的全连接网络，就不能这样做了；
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10]) # 输入真实的label
# 使用Softmax Regression模型处理多分类任务
# 工作原理：将可以判定为某类的特征相加，然后将这些特征转化为判定是这一类的概率。
#          比如某个像素的灰度值很大代表很可能是数字n时，这个像素的权重就很大。
y = tf.nn.softmax(tf.matmul(x, W) + b)  # tf.nn包含了大量神经网络的组件；tf.matmul是矩阵乘法函数


# 3 定义损失函数，并设置梯度下降的方式（优化算法）
# Tensorflow在训练时会根据计算图自动求导并进行梯度下降；通常loss使用cross-entropy作为loss function；
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # cross_entropy ; tf.reduce_mean用来对每个batch数据结果求均值
# 定义优化算法，这里以SGD为例
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# 4 训练
# 首先初始化
init = tf.global_variables_initializer()
# clw note：也可以不用下面的with tf.Session() as sess:的方式，而用sess = tf.InteractiveSession()
# tf.InteractiveSession()是一种交互式的session方式，它让自己成为了默认的session，
# 也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来，这就是默认的好处。
# 这样的话就是run()和eval()函数可以不指明session啦。
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoches):
        avg_cost = 0
        #total_batch = int(num_examples / batch_size)  # clw note：这里是不是需要向上取整？
        total_batch = math.ceil(num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #取128个样本构成一个mini-batch，并feed给placeholder
            # 然后调用train_step对这些样本进行训练
            #optimizer.run({x:batch_xs, y_:batch_ys})
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs, y_:batch_ys})
            avg_cost += c / total_batch
        # 显示每次迭代的结果
        if (epoch+1) % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print('Optimization Finished!')

# 5 测试
# 验证集
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # tf.argmax是从一个tensor中寻找最大值对应的index
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast将之前输出的bool值转为float32再求平均
    print('clw:验证集accuracy = ', accuracy.eval({x:mnist.validation.images, y_:mnist.validation.labels}))
# 测试集
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # tf.argmax是从一个tensor中寻找最大值对应的index
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast将之前输出的bool值转为float32再求平均
    print('clw:测试集accuracy = ', accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
    # clw note：对上面的eval方法，如果你有一个tf.Tensor对象t，在使用t.eval()时，等价于：tf.get_default_session().run(t).
    # 注意使用eval方法的时候，一定要保证之前已经定义了默认的session，或者eval处于with tf.Session() as sess:的内部！！！
    # 每次使用 eval 和 run时，都会执行整个计算图，为了获取计算的结果，将它分配给tf.Variable，然后获取。

    # 预测结果可视化，比如这里只看前10张图片以及预测的结果
    for i in range(10):
        image = mnist.test.images[i].reshape(-1, 28) # 其实就是28*28
        y_predict = sess.run(tf.argmax(y_, 1)[i], {x:mnist.test.images, y_:mnist.test.labels})
        print('clw：预测的结果为', y_predict)
        plt.imshow(image)
        plt.axis('off') # 去掉x，y坐标轴以及刻度信息
        plt.show()

print('clw:--------------------end!--------------------')

### 总结
# 上面是用Tensorflow实现一个简单的机器学习算法Softmax Regression，可以算是一个没有隐藏层的“最浅层神经网络”
# 整个流程分为4个步骤：
#  (1) 加载数据，数据预处理，设置超参
# （2）搭建网络结构，即神经网络forward时的计算
# （3）定义loss，选择优化器，并指定优化器优化loss
# （4）迭代地对数据训练
# （5）在验证集/测试集上对准确率进行测评