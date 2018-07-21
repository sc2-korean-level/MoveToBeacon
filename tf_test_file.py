import tensorflow as tf


a = tf.placeholder(tf.float32, [None, 3])
b = tf.log(a)
c = tf.reduce_sum(b, axis=1)

xx = [[1,2,3], [4,5,6]]

sess = tf.Session()

aaaa = sess.run(b, feed_dict={a: xx})
bbbb = sess.run(c, feed_dict={a: xx})
print(aaaa)
print(bbbb)