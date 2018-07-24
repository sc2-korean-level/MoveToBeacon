import tensorflow as tf
import numpy as np

sess = tf.Session()

rewards = np.genfromtxt('4wayBeacon_a2c/reward.csv')

r = tf.placeholder(tf.float32)
rr = tf.summary.scalar('reward', r)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tensorboard_data/a2c', sess.graph)

for i, reward in enumerate(rewards):
    summary = sess.run(merged, feed_dict={r: reward})
    writer.add_summary(summary, i)
    print(i)