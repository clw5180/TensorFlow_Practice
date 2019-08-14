#--coding--:utf-8
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

USE_TENSORBOARD = True

# 1、获取数据，数据预处理，设置超参数
housing = fetch_california_housing()
m, n = housing.data.shape
print("数据集:{}行, {}列".format(m,n))

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# 1、数据预处理、设置超参数
n_epochs = 20
learning_rate = 0.001

batch_size = 128
n_batches = int(np.ceil(m / batch_size))
#有放回的随机抽取数据
def fetch_batch(epoch, batch_index, batch_size):
    #定义一个随机种子
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch


# 2、搭建网络
# 使用mini-batch方式
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y  # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")  # Mean Squared Error


# 3、定义损失函数，并设置梯度下降的方式（优化算法）
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # not shown
training_op = optimizer.minimize(mse)  # not shown


# 4、训练
init = tf.global_variables_initializer()

if USE_TENSORBOARD:
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    mse_summary = tf.summary.scalar('MSE', mse) # 第一行创建一个节点，这个节点将求出 MSE 值并将其写入 TensorBoard 兼容的二进制日志字符串（称为摘要）中。
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


#开始运行
with tf.Session() as sess:
    sess.run(init)
#每次都抽取新的数据做训练
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            loss = mse.eval(feed_dict={X: X_batch, y: y_batch})

            # print acc_train,total_loss
            print("epoch:", epoch+1, "batch_id:", batch_index, "Batch loss:", loss)

            if USE_TENSORBOARD:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)

#最终结果
    best_theta = theta.eval()  # clw note：这里theta是个tf.Variable类型，所以括号内不用再加feed_dict了
    print('best theta:', theta.eval())
    if USE_TENSORBOARD:
        file_writer.close()