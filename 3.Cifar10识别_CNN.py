import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

#cifar10:https://zhuanlan.zhihu.com/p/27017189;
#tf:https://blog.csdn.net/lenbow/article/details/52152766;
# cifar10：60000张32*32的彩色图片，训练集50000，测试集10000

max_steps = 50
batch_size = 128
K = 3  # for tf.nn.in_top_k - Says whether the label_holder are in the top `K` of logits.
       # 比如10个类别只要得分最高的K个里面含有label，就算预测正确
#data_dir = './cifar10_data/cifar-10-batches-bin'

# 权重初始化函数 add_to_collection:https://blog.csdn.net/uestc_c2_403/article/details/72415791;
# 这里给weight加一个L2 loss
# 一般来说，L1正则会造成稀疏的特征，大部分无用特征的权重会被置为0，而L2正则会让特征的权重不过大，使得特征的权重比较平均
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)  # 把weight_loss集中到losses中以计算总体loss
    return var
	
# loss搭建：
# cross_entropy:https://zhuanlan.zhihu.com/p/27842203;
# tf.add_n:https://blog.csdn.net/uestc_c2_403/article/details/72808839;
def loss(logits, labels):
#      """Add L2Loss to all the trainable variables.
#      Add summary for "Loss" and "Loss/avg".
#      Args:
#        logits: Logits from inference().
#        labels: Labels from distorted_inputs or inputs(). 1-D tensor
#                of shape [batch_size]
#      Returns:
#        Loss tensor of type float.
#      """
#      # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
  

###cifar10.maybe_download_and_extract()  # clw note：该函数已弃用，解决方法详见README文件

# 数据集准备：
# 生成训练的数据集（返回已经封装好的tensor）
# 数据增强操作包括随机的水平翻转，随机剪切一块24*24大小的图片，设置随机亮度和对比度，以及数据标准化
# 数据增强的操作会耗费大量CPU时间，因此distorted_inputs使用了16个独立的线程来加速任务，函数内部会产生线程池，
# 在需要的时候会通过TensorFlow queue进行调度
#images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_train, labels_train = cifar10_input.distorted_inputs(batch_size=batch_size)

# 生成测试的数据集
# 测试数据不需要太多处理，只需要裁剪图片正中间的24*24大小的区块，并进行数据标准化操作
#images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, batch_size=batch_size)
# placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 网络搭建：
# 第一层：卷积层+池化层+LRN层
# 第一层卷积层卷积核大小5x5,颜色通道3,卷积核数64,标准差0.05,w1=0;con2d卷积，步长stride=1，padding为same；bias全为0；非线性化激活函数：ReLU；最大池化尺寸3x3，步长2x2；使用LRN优化ReLU
# lrn:https://blog.csdn.net/banana1006034246/article/details/75204013;
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二层：卷积层+LRN层+池化层
# 第二层卷积层卷积核大小5x5,颜色通道3,卷积核数64,标准差0.05,w1=0;con2d卷积，步长stride=1，padding为same；bias全为0.1；非线性化激活函数：ReLU；最大池化尺寸3x3，步长2x2；使用LRN优化ReLU
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第三层：全连接层
# 第三层：全连接层隐含节点数384，标准差0.04，bias=0.1，weight loss=0.004防止过拟合
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第四层：全连接层
# 第四层：全连接层隐含节点数192，标准差0.04，bias=0.1，weight loss=0.004防止过拟合
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 第五层：全链接层，输出层
# 第五层：全连接层输出层节点数10，标准差1/192.0，bias=0.0
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
print('clw:查看logits的shape： ', logits.get_shape())  # shape是(128, 10)，代表了batch_size个样本中，每个样本在10个类别的得分，比如
                                        # 对于某个样本，有[1.1 4.3 10.5 2.2 4.3 ....  ]
print('clw:查看label_holder的shape： ', label_holder.get_shape())  # shape是(128, 1)，代表了batch_size个样本中，每个样本该类别的标签，比如
                                          # 对于某个样本标签是3（比如代表cat）；之后会判断上面得分最高的k个值对应的index，
                                        # 是否有等于这个label的，等于则说明判断正确了。

loss = loss(logits, label_holder)
# 优化器Adam Optimizer
# 常用优化器：https://blog.csdn.net/xierhacker/article/details/53174558；https://blog.csdn.net/qiurisiyu2016/article/details/80383812；
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, K)  # Says whether the label_holder are in the top `K` of logits.

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners() # 图片数据增强的线程队列，这里一共使用了16个线程

# 开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss = %.2f(%.1f examples/sec; %.3f sec/batch)')
        # 在GTX1080上，每秒钟可以训练大约1800个样本（第2个参数），如果batch_size为128，每个batch大约需要0.066s
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch)) # loss一开始大约4.6，经过3000步之后降到1.0附近


# 开始测试
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size)) # ceil意思是天花板，和地板除正好反过来：回大于或等于一个给定数字的最小整数。
true_count = 0
total_sample_count = num_iter * batch_size
step = 0

while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    ### clw modify：调试用
    if step == num_iter - 1:
        print('clw:len(predictions) = ', len(predictions))
        print('clw:predictions[0].shape = ', predictions[0].shape)
        print('clw:predictions', predictions)
    #############################################
    true_count += np.sum(predictions)
    step += 1
precision = true_count / total_sample_count
print('precision @top_%d_op = %.3f' %(K, precision))
