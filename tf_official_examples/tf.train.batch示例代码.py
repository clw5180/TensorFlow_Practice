import tensorflow as tf
import numpy as np
import math

num_examples = 20
batch_size = 5
num_epochs = 3  # 1个epoch要迭代math.ceil(num_examples / batch_size) = 10次，也就是10个step，这里设置训练3轮，也就是30个step

def generate_data():
    num = num_examples
    label = np.asarray(range(0, num)) # 将结构数据转化为ndarray
    images = np.random.random([num, 5, 5, 3]) # 随机生成num张，尺寸为5*5, 3通道的假图片
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return label, images


def get_batch_data():
    label, images = generate_data()
    print('clw:label = ', label)
    images = tf.cast(images, tf.float32) # Casts a tensor to a new type
    label = tf.cast(label, tf.int32)

    # 官方：Produces a slice of each Tensor in tensor_list. (deprecated)
    # tf.train.slice_input_producer是一个tensor生成器，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列
    # images是一个tensor，shape是(20,5,5,3)，label也是一个tensor，shape是(20,)
    # 返回的也是一个list，shape是(5,5,3)和()；相当于从batch_size=20的这批数据中取出1张图片和对应的标签
    # 另外要注意默认shuffle=True，这里手动改成False
    # num_epochs = N，表示生成器只能遍历tensor列表N次；否则可以无限次遍历tensor列表，需要在其他地方手动设置最大训练轮数
    input_queue = tf.train.slice_input_producer([images, label], num_epochs = num_epochs, shuffle=False)
    print('clw:input_queue[0].get_shape() = ', input_queue[0].get_shape())
    print('clw:input_queue[0].get_shape() = ', input_queue[1].get_shape())

    # 官方：Creates batches of tensors in tensors. (deprecated)
    # 自注：用上面的方法把样本个数为m的image或label数据打散成单个样本，然后在这里组成batch
    # tf.train.batch是一个tensor队列生成器，作用是按照给定的tensor顺序，
    # 把batch_size个tensor推送到文件队列，作为训练一个batch的数据，等待tensor出队执行计算。
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    return image_batch, label_batch


image_batch, label_batch = get_batch_data()
with tf.Session() as sess:
    # initializer for num_epochs
    tf.local_variables_initializer().run()

    # TensorFlow Session对象是多线程的，所以多个线程可以轻松地使用相同的Session并并行运行op。但是，实现一个驱动线程
    # 的Python程序并不容易。所有线程必须能够一起停止，异常必须被捕获和报告，队列必须在停止时正确关闭。TensorFlow提供
    # 了两个类来帮助线程驱动：tf.train.Coordinator和tf.train.QueueRunner。这两个类被设计为一起使用，协调器类帮助
    # 多个线程一起停止，并向等待其停止的程序报告异常。QueueRunner类用于创建多个线程，用于协调在同一队列中放入张量。
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)  # 要真正将文件放入文件名队列，还需要调用tf.train.start_queue_runners()函数
                                                         # 来启动执行文件名队列填充的线程，之后计算单元才可以把数据读出来，
                                                         # 否则文件名队列为空的，计算单元就会处于一直等待状态，导致系统阻塞。
    step = 1
    try:
        while not coord.should_stop():
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            print('clw: step = ', step)
            for j in range(batch_size):
                print(image_batch_v[j].shape, label_batch_v[j])
                if j == batch_size - 1: # clw note：便于观察结果
                    print('')
            step += 1
    except tf.errors.OutOfRangeError:  # 在跑了超过num_epochs轮之后，如果再尝试读入，系统由于检测到了“结束”，
                                    # 就会自动抛出一个异常（OutOfRange）。外部捕捉到这个异常后就可以结束程序了
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)