import tensorflow as tf

a = tf.placeholder(tf.float32, [None, 3])
b = tf.placeholder(tf.float32, [None, 3])

c = tf.reduce_sum(a * b, axis=1)

d = tf.placeholder(tf.float32, [None, 4])
e = tf.placeholder(tf.float32, [None, 4])

f = tf.reduce_sum(d * e, axis=1)

g = c * f


aa = [[0.3, 0.2, 0.5], [0.2, 0.3, 0.5], [0.3, 0.2, 0.5]]
bb = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
cc = [[0.2, 0.2, 0.1, 0.5], [0.2, 0.2, 0.1, 0.5], [0.2, 0.2, 0.1, 0.5]]
dd = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]


sess = tf.Session()

result1 = sess.run(c, feed_dict={a: aa, b: bb})
result2 = sess.run(f, feed_dict={d: cc, e: dd})
result3 = sess.run(g, feed_dict={a: aa, b: bb, d: cc, e: dd})
print(result1)
print(result2)
print(result3)