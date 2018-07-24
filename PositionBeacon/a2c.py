import tensorflow as tf
import numpy as np

def disconut_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r

class a2c:
    def __init__(self, sess, exp_rate):
        self.sess = sess
        self.state_size = 16*16*2
        self.action_size = 3
        self.exp_rate = exp_rate

        self.X = tf.placeholder(tf.float32, [None, 16*16*2], name='X')
        self.a = tf.placeholder(tf.float32, [None, 3], name='a')
        self.sp = tf.placeholder(tf.float32, [None, 16*16], name='sp')
        self.r = tf.placeholder(tf.float32, [None, 1], name='r')
        self.v_ = tf.placeholder(tf.float32, [None, 1], name='v_')
        self.actor, self.spatial, self.critic = self._bulid_net()

        self.td_error = self.r + 0.99 * self.v_ - self.critic
        print(self.td_error)
        self.closs = tf.square(self.td_error)

        pi_a_s = tf.reduce_sum(self.actor * self.a, axis=1)
        log_pi_a_s = tf.log(pi_a_s)
        self.aloss = log_pi_a_s * tf.reshape(self.td_error, [-1])

        self.loss = -self.aloss -self.closs
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def learn(self, state, next_state, reward, action, spatial):
        v_ = self.sess.run(self.critic, feed_dict={self.X: next_state})
        _ = self.sess.run(self.train_op,
                          feed_dict={self.X: state, self.v_: v_, self.r: reward, self.a: action, self.spatial: spatial})

    def _bulid_net(self):
        reshape = tf.reshape(self.X, [-1, 16, 16, 2])
        layer_1 = tf.layers.conv2d(inputs=reshape, filters=16, kernel_size=[5, 5], activation=tf.nn.relu, padding='SAME')
        layer_2 = tf.layers.conv2d(inputs=layer_1, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')

        reshape_ = tf.reshape(layer_2, [-1, 16*16*32])
        action_1 = tf.layers.dense(inputs=reshape_, units=256, activation=tf.tanh)
        action = tf.layers.dense(inputs=action_1, units=3, activation=tf.nn.softmax)

        spatial_1 = tf.layers.conv2d(inputs=layer_2, filters=1, kernel_size=[1, 1], activation=tf.tanh)
        reshape__ = tf.reshape(spatial_1, [-1, 16*16])
        spatial = tf.layers.dense(inputs=reshape__, units=16*16, activation=tf.nn.softmax)

        critic_1 = tf.layers.dense(inputs=reshape_, units=256, activation=tf.tanh)
        critic = tf.layers.dense(inputs=critic_1, units=1, activation=None, trainable=True)

        return action, spatial, critic

    def choose_action(self, s):
        action, spatial = self.sess.run([self.actor, self.spatial], feed_dict={self.X: [s]})
        return action, spatial

def disconut_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r