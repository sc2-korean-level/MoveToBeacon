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
        self.state_size = 2
        self.action_size = 4
        self.exp_rate = exp_rate

        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.a = tf.placeholder(tf.float32, [None, self.action_size])
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.v_ = tf.placeholder(tf.float32, [None, 1])
        self.actor, self.critic = self._bulid_net()

        self.td_error = self.r + 0.99 * self.v_ - self.critic
        self.closs = tf.square(self.td_error)
        self.train_cop = tf.train.AdamOptimizer(0.0001).minimize(self.closs)

        self.log_lik = self.a * tf.log(self.actor)
        self.log_lik_adv = self.log_lik * self.td_error
        self.exp_v = tf.reduce_mean(tf.reduce_sum(self.log_lik_adv, axis=1))
        self.entropy = -tf.reduce_sum(self.actor * tf.log(self.actor))
        self.obj_func = self.exp_v + self.exp_rate * self.entropy
        self.loss = -self.obj_func
        self.train_aop = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def learn(self, state, next_state, reward, action):
        v_ = self.sess.run(self.critic, feed_dict={self.X: next_state})
        _, _ = self.sess.run([self.train_cop, self.train_aop],
                          feed_dict={self.X: state, self.v_: v_, self.r: reward, self.a: action})

    def _bulid_net(self):
        layer_1 = tf.layers.dense(inputs=self.X, units=64, activation=tf.tanh)
        layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
        layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.tanh)
        layer_4 = tf.layers.dense(inputs=layer_3, units=4, activation=tf.tanh)
        actor = tf.layers.dense(inputs=layer_4, units=self.action_size, activation=tf.nn.softmax)

        layer_1 = tf.layers.dense(inputs=self.X, units=64, activation=tf.tanh)
        layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
        layer_3 = tf.layers.dense(inputs=layer_2, units=30, activation=tf.tanh)
        critic = tf.layers.dense(inputs=layer_3, units=1, activation=None)

        return actor, critic

    def choose_action(self, s):
        act_prob = self.sess.run(self.actor, feed_dict={self.X: [s]})
        action = np.random.choice(self.action_size, p=act_prob[0])
        return action

def disconut_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for i in reversed(range(len(r))):
        running_add = running_add * 0.99 + r[i]
        discounted_r[i] = running_add
    discounted_r = (discounted_r - discounted_r.mean())/(discounted_r.std() + 1e-7)
    return discounted_r