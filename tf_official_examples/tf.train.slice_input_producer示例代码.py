# 参考：https://blog.csdn.net/dcrmg/article/details/79776876
# 在N个epoch的文件名最后是一个结束标志，当tf读到这个结束标志的时候，会抛出一个 OutofRange 的异常，
# 外部捕获到这个异常之后就可以结束程序了。而创建tf的文件名队列就需要使用到 tf.train.slice_input_producer函数。

import tensorflow as tf

images = ['img1', 'img2', 'img3', 'img4', 'img5']
labels = [1, 2, 3, 4, 5]

epoch_num = 2

f = tf.train.slice_input_producer([images, labels], num_epochs=epoch_num, shuffle=False)

with tf.Session() as sess:
    # initializer for num_epochs
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(20):
        k = sess.run(f)
        print
        '************************'
        print(i, k)

    coord.request_stop()
    coord.join(threads)