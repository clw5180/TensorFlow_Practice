# 参考https://zhuanlan.zhihu.com/p/27238630 何之源

# tf.train.string_input_producer - 文件名队列
# 若设置shuffle=False，则每个epoch内，数据还是按照A、B、C的顺序进入文件名队列，这个顺序不会改变：
# 在tensorflow中，内存队列不需要我们自己建立，我们只需要使用reader对象从文件名队列中读取数据就可以了.

import tensorflow as tf

num_epochs = 2

# 新建一个Session
with tf.Session() as sess:
    # 我们要读三幅图片
    filename = ['C:/Users/Administrator/Desktop/images/1.png',
                'C:/Users/Administrator/Desktop/images/2.png',
                'C:/Users/Administrator/Desktop/images/3.png']
    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=num_epochs)
    # reader从文件名队列中读数据。对应的方法是reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之后，才会开始填充队列
    threads = tf.train.start_queue_runners(sess=sess)
    i = 1
    try:
        while True:
            # 获取图片数据并保存
            image_data = sess.run(value)
            print('clw: i = ', i)
            with open('./test_%d.png' % i, 'wb') as f:
                f.write(image_data)
            i += 1
    except tf.errors.OutOfRangeError:  # 在跑了超过num_epochs轮之后，如果再尝试读入，系统由于检测到了“结束”，
                                    # 就会自动抛出一个异常（OutOfRange）。外部捕捉到这个异常后就可以结束程序了
        print("done")