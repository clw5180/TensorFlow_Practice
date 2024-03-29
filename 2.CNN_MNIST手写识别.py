from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# 使用截断的正态分布来初始化权重
# 给权重一些随机的噪声，打破完全对称；比如截断的正态分布噪声，标准差0.1,
# 因为使用ReLU，故给偏置增加一些小的正值如0.1，避免失活节点（dead neurons）

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# tf.nn.conv2d是TensorFlow中的2维卷积函数，参数中x是输入，W是卷积的参数，
# 比如[5,5,1,32]，代表h, w, n_c , n_H
# 因为希望整体上缩小图片尺寸，注意pooling层的strides=2
# strides在官方定义中是一个一维具有四个元素的张量，其规定前后必须为1，
# 所以我们可以改的是中间两个数，中间两个数分别代表了水平滑动和垂直滑动步长值。
# 而对于padding：
# 吴恩达课程：“SAME”表示超过边界用0填充，使得输出保持和输入相同的大小，“VALID”表示不超过边界，通常会丢失一些信息。
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # W中包含了filter_size的信息

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME') # pooling层无训练参数，只需设置ksize


# 1 定义输入的placeholder；因为CNN会利用空间结构信息，因此要将1D的784维输入向量转为28*28的2D的图片结构；
x = tf.placeholder(tf.float32, [None, 784]) # None代表输入的样本个数是不确定的，784代表每个样本都对应一个784维的向量
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1]) # 因为只有1个颜色通道，且样本数量不固定，用-1表示，故最终尺寸为[-1, 28, 28, 1]

# 2 定义卷积层，注意先初始化参数W和b
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 尺寸由28*28变成14*14

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 尺寸由14*14变成7*7，即输出tensor尺寸为7*7*64

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 使用一个Dropout层来减轻过拟合：通过一个placeholder传入keep_prob比率来控制的，见Chapter 4
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 3 连接一个Softmax层，的到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 4 定义损失函数，这里用Adam，并给予一个较小的学习率1e-4
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 5 定义测评准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 6 开始训练过程
tf.global_variables_initializer().run()
for i in range(500):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" %(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test accuracy %g" %accuracy.eval(feed_dict={x:mnist.test.images[0:1000], y_:mnist.test.labels[0:1000], keep_prob:1.0}))
### 自注：上面一句如果不写[0:1000]，则会一次性加载10000张图片，我的2G显存电脑会OOM




