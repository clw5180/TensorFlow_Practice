import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# 1 加载数据集
mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)

# 2 数据预处理（28*28的图，展成长度为784一维向量；这里丢弃了二位空间的信息，因此效果肯定不如CNN
#   训练输入是55000*784的Tensor，对应label是55000*10的Tensor，这里对10个种类进行one-hot编码
print('clw：mnist.train.images.shape = ', mnist.train.images.shape, 'mnist.train.labels.shape = ', mnist.train.labels.shape)
print('clw：mnist.test.images.shape = ', mnist.test.images.shape, 'mnist.test.labels.shape = ',mnist.test.labels.shape)
print('clw：mnist.validation.images.shape = ', mnist.validation.images.shape, 'mnist.validation.labels.shape = ', mnist.validation.labels.shape)


# 3 初始化参数
x = tf.placeholder(tf.float32, [None, 784]) # None代表输入的样本个数是不确定的，784代表每个样本都对应一个784维的向量
W = tf.Variable(tf.zeros([784, 10])) # 注意这里可以全部初始化为0，但对于CNN或者比较深的全连接网络，就不能这样做了；
b = tf.Variable(tf.zeros([10]))


# 4 使用Softmax Regression模型处理多分类任务
#   工作原理：将可以判定为某类的特征相加，然后将这些特征转化为判定是这一类的概率。
#            比如某个像素的灰度值很大代表很可能是数字n时，这个像素的权重就很大。
y = tf.nn.softmax(tf.matmul(x, W) + b)  # tf.nn包含了大量神经网络的组件；tf.matmul是矩阵乘法函数


# 5 定义损失函数，Tensorflow在训练时会根据计算图自动求导并进行梯度下降；通常loss使用cross-entropy作为loss function；
y_ = tf.placeholder(tf.float32, [None, 10]) # 输入真实的label
                                            # tf.reduce_mean用来对每个batch数据结果求均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# 6 定义优化算法，这里以SGD为例
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

# 7 训练
#sess = tf.InteractiveSession()  # 创建一个会话，在会话中对变量进行初始化操作
#init = tf.global_variables_initializer() # 使用Tensorflow的全局参数初始化器tf.global_variables_initializer，执行它的run方法
#sess.run(init)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #取100个样本构成一个mini-batch，并feed给placeholder
    # 然后调用train_step对这些样本进行训练
    train_step.run({x:batch_xs, y_:batch_ys})


# 8 验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # tf.argmax是从一个tensor中寻找最大值对应的index
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast将之前输出的bool值转为float32再求平均
print('clw:测试集accuracy = ', accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))

### 总结
# 上面是用Tensorflow实现一个简单的机器学习算法Softmax Regression，可以算是一个没有隐藏层的“最浅层神经网络”
# 整个流程分为4个步骤：
# 1、定义算法公式，即神经网络forward时的计算
# 2、定义loss，选择优化器，并指定优化器优化loss
# 3、迭代地对数据训练
# 4、在验证集/测试集上对准确率进行测评


# 可以看一下数据集的图片，比如这里只看前10章以及预测的结果
for i in range(10):
    image = mnist.test.images[i].reshape(-1, 28) # 其实就是28*28
    y_predict = sess.run(tf.argmax(y_, 1)[i], {x:mnist.test.images, y_:mnist.test.labels})
    print('clw：预测的结果为', y_predict)
    plt.imshow(image)
    plt.axis('off') # 去掉x，y坐标轴以及刻度信息
    plt.show()


print('clw:--------------------end!--------------------')