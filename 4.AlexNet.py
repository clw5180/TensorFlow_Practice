from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

### 功能：显示网络
# 自注：tf.shape()：一般用于获取维度不确定的变量的维度，返回的是一个tensor。要想知道是多少，必须通过sess.run()
# 而get_shape()只用于获取tensor的维度；返回的是一个元组，返回的是元组，不能放到sess.run()里面，因为这里面只能放operation和tensor；
# 通过as_list()的操作转换成list，其实这里不转也行
def print_activations(t):
    #print(t.op.name, ' ', t.get_shape().as_list())
    print(t.op.name, ' ', t.get_shape())


### 功能：搭建AlexNet的卷积部分
def inference(images):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope: # 可以将scope内生成的variable自动命名为conv1/xxx，便于区分不同卷积层之间的组件
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1) # 打印conv1层最后输出的tensor的维度
        parameters += [kernel, biases] # 将conv1层的可训练参数kernel、biases添加到parameters中


  # pool1
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1') # 除了AlexNet，其他网络基本都放弃了LRN，
                                                                                    # 主要是效果不明显，且速度大幅降低
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1') # 最大值池化，将3*3大小的像素块降为1*1的像素；VALID表示取样时不超过边框，
                                         # 不像SAME模式那样可以填充边界外的点
    print_activations(pool1)

  # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

  # pool2
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

  # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

  # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

  # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

  # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)

    # 至此，inference函数完成了，使用AlexNet进行训练或预测时，还需要添加3个全连接层，隐含结点数为4096、4096、1000
    # 计算量很小，耗时很少
    return pool5, parameters


### 功能：评估AlexNet每轮计算时间
def time_tensorflow_run(session, target, info_string):
#  """Run the computation to obtain the target tensor and print timing stats.
#
#  Args:
#    session: the TensorFlow session to run the computation under.
#    target: the target Tensor that is passed to the session's run() function.
#    info_string: a string summarizing this run, to be printed with the stats.
#
#  Returns:
#    None
#  """
    num_steps_burn_in = 10  # 定义预热轮数，因为头几轮迭代有显存加载，cache命中等问题，可以跳过，只考量10轮迭代之后的计算时间
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                # 结论：i5 4210比GTX1080慢了大概50倍；
                # 对于正向传播：
                # 书上用GTX1080，在有LRN层每轮迭代时间是0.026s，没有则为0.007s
                # 我这里用i5 4210，有LRN层每轮迭代时间是1.026s
                # 对于反向传播：
                # 书上用GTX1080，在有LRN层每轮迭代时间是0.078s，没有则为0.025s
                # 我这里用i5 4210，有LRN层每轮迭代时间是3.938s

                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))


### 功能：主函数，这里不用数据集训练，只用随机图片数据测试前馈和反馈计算的耗时；
def run_benchmark():
#  """Run the benchmark on AlexNet."""
    with tf.Graph().as_default():
    # Generate some dummy images.
        image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
        images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    # Build a Graph that computes the logits predictions from the
    # inference model.
        pool5, parameters = inference(images)

    # Build an initialization operation.
        init = tf.global_variables_initializer()

    # Start running operations on the Graph.
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        sess.run(init)

    # Run the forward benchmark.
        time_tensorflow_run(sess, pool5, "Forward")

    # Add a simple objective so we can calculate the backward pass.
        objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.
        grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
        time_tensorflow_run(sess, grad, "Forward-backward")


run_benchmark()
