import numpy as np
import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, temp=0.1):

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 16*16*2], name='obs')
            reshape = tf.reshape(self.obs, [-1, 16, 16, 2])
            layer_1 = tf.layers.conv2d(inputs=reshape, filters=16, kernel_size=[5, 5], strides=[1, 1], padding='SAME', activation=tf.nn.relu)
            layer_2 = tf.layers.conv2d(inputs=layer_1, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation=tf.nn.relu)
            with tf.variable_scope('policy_net'):
                reshape_spatial = tf.reshape(layer_2, [-1, 16*16*32])
                
                dense_1 = tf.layers.dense(inputs=reshape_spatial, units=256, activation=tf.nn.relu)
                self.act_probs = tf.layers.dense(inputs=dense_1, units=3, activation=tf.nn.softmax)
                self.spatial_probs = tf.layers.dense(inputs=dense_1, units=16*16, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                dense_2 = tf.layers.dense(inputs=dense_1, units=64, activation=tf.nn.relu)
                self.v_preds = tf.layers.dense(inputs=dense_2, units=1, activation=None, trainable=True, kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.scope = tf.get_variable_scope().name

    def act(self, obs):
        return tf.get_default_session().run([self.act_probs, self.spatial_probs, self.v_preds], feed_dict={self.obs: [obs]})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
