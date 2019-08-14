import tensorflow as tf

with tf.variable_scope('f2'):
    # a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a1 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a1')
    a2 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a1')
    a5 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))

with tf.variable_scope('f2', reuse=True):
    a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))

    a4 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')

print(a1.name)
print(a2.name)
print(a1 == a2)
print(a5.name)
print(a3.name)
print(a4.name)