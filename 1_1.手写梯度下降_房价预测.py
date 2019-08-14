#--coding--:utf-8
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

USE_TENSORBOARD = True
SAVE_MODEL = False
RESTORE_MODEL = False

# 1、获取数据，数据预处理，设置超参数
housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行, {}列".format(m,n))

# StandardScaler默认就做了方差归一化，和均值归一化，这两个归一化的目的都是为了更快的进行梯度下降
# 当使用梯度下降时，首先要对输入特征向量进行归一化，否则训练可能要慢得多。
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data] # np.c_是按列连接两个矩阵，要求行数相等。
                                                                            # 比如a=[1,2,3]  b=[4,5,6]，合并之后就是[1,2,3,4,5,6]
                                                                            # 这里加了一个全为1的维度，因为这里把偏置b也放到了theta中，
                                                                            # 原来的W^TX + b变成了theta^TX，因此多了一个全1的列

# 设置超参数
n_epochs = 500
learning_rate = 0.01

# 2、搭建网络
X = tf.constant(scaled_housing_data_plus_bias, dtype = tf.float32, name = "X") # 当数据量达到几G，几十G时，使用constant直接导入数据显然是不现实的，因此可以看1_2版本程序，使用placeholder
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")  # 注意这里X的维度是(m, n)所以是X乘theta；吴恩达书里是(n, m)，所以还是看
error = y_pred - y  # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")  # Mean Squared Error


# 3、定义损失函数，并设置梯度下降的方式（优化算法）
# 方法（1）：手动计算梯度，迭代的方式求权重；从代价函数（MSE）中利用数学公式推导梯度
# （但如果要用深层神经网络做这个事情，这将变得很复杂，而且容易出错）
# 注：这里用了tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)函数，完成了
#    将value赋值给ref的作用。其中：ref 必须是tf.Variable创建的tensor，如果ref=tf.constant()会报错！
#gradients = 2/m * tf.matmul(tf.transpose(X), error)
#training_op = tf.assign(theta, theta - learning_rate * gradients)

# 方法（2）：自动计算梯度，迭代的方式求权重；结果和手动一致
#gradients = tf.gradients(mse, [theta])[0]  # 使用一个op（这里是MSE）和一个变量列表（这里只是theta），它创建一个ops列表（每个变量一个）来计算op的梯度变量
#training_op = tf.assign(theta, theta - learning_rate * gradients)

# 方法（3）：使用优化器；如果使用GradientDescentOptimizer，结果和手动一致
# clw note：看起来RMSPropOptimizer和Adam还有MomentumOptimizer(momentum=0.9)效果比较好，AdagradOptimizer直接跑飞
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # not shown
training_op = optimizer.minimize(mse)  # not shown
# 经验之谈：
# SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠
# 如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。
# Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。
# 在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果

######################################
# 注：这里也可以用最小二乘法求权重，一步到位，省去了定义损失函数，并设置梯度下降的方式和训练（迭代）的步骤，
#     但只能解决简单问题，是一种线性的模型。
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)  # clw note：只有方阵才有性质(AB)^-1 = A^-1B^-1
# with tf.Session() as sess:
#     theta_value = theta.eval()
# print(theta_value)
#######################################



# 4、训练
init = tf.global_variables_initializer()

# 另：节点的保存与恢复
if SAVE_MODEL:
    saver = tf.train.Saver()  # 在构造阶段结束（创建所有变量节点之后）后创建一个保存节点; 在执行阶段只要调用它的save()方法就可以保存模型

# 另：tensorboard可视化
if USE_TENSORBOARD:
    from datetime import datetime
    # now是本地时间，可以认为是你电脑现在的时间，还是要用这个！
    # utcnow是世界时间（时区不同，所以这两个是不一样的）
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    mse_summary = tf.summary.scalar('MSE', mse) # 第一行创建一个节点，这个节点将求出 MSE 值并将其写入 TensorBoard 兼容的二进制日志字符串（称为摘要）中。
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  # 创建一个FileWriter，您将用它来将摘要写入日志目录中的日志文件中。
                                                                         # 第一个参数指示日志目录的路径（在本例中为tf_logs/run-20160906091959/，相对于当前目录）
                                                                         # 第二个（可选）参数是您想要可视化的图形。

with tf.Session() as sess:
    if RESTORE_MODEL:
        saver.restore(sess, "./my_model_final.ckpt")  # 恢复模型同样容易：在构建阶段结束时创建一个保存器，在执行阶段的开始调用
                                                   # saver节点的restore()方法，而不再使用init节点初始化变量（sess.run(init)）
    sess.run(init)
    for epoch in range(n_epochs):
        if (epoch+1) % 10 == 0:
            print("Epoch:", '%04d' % (epoch+1), "MSE =", mse.eval())
            if SAVE_MODEL:
                save_path = saver.save(sess, "./my_model_%04d.ckpt" % (epoch+1))  # checkpoint every 100 epochs
            if USE_TENSORBOARD:
                summary_str  = mse_summary.eval()
                file_writer.add_summary(summary_str , epoch)

        sess.run(training_op) # 或者training_op.eval()
    best_theta = theta.eval()
    print('best theta:', theta.eval())

    if USE_TENSORBOARD:
        file_writer.close()

    if SAVE_MODEL:
        save_path = saver.save(sess, "./my_model_final.ckpt")

# 详见https://github.com/apachecn/hands-on-ml-zh/blob/master/docs/9.%E5%90%AF%E5%8A%A8%E5%B9%B6%E8%BF%90%E8%A1%8C_TensorFlow.md



