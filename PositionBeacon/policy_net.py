import gym
import numpy as np
import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, temp=0.1):

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 16*16*2], name='obs')
            reshape = tf.reshape(self.obs, [-1, 16, 16, 2])
            layer_1 = tf.layers.conv2d(inputs=reshape, filters=16, kernel_size=[5, 5], activation=tf.nn.relu, padding='SAME')
            layer_2 = tf.layers.conv2d(inputs=layer_1, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
            
            with tf.variable_scope('policy_net'):
                reshape_ = tf.reshape(layer_2, [-1, 16*16*32])
                dense_1 = tf.layers.dense(inputs=reshape_, units=256, activation=tf.nn.relu)
                self.action_policy_prob = tf.layers.dense(inputs=tf.divide(dense_1, temp), units=3, activation=tf.nn.softmax)
            
                spatial_policy_prob = tf.layers.conv2d(layer_2, filters=1, kernel_size=[1, 1], activation=tf.nn.relu)
                reshape_conv = tf.reshape(spatial_policy_prob, [-1, 16*16])
                self.spatial_policy_prob = tf.layers.dense(inputs=reshape_conv, units= 16*16, activation=tf.nn.softmax)
            
            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=64, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=30, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            #print(tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs}))
            action, spatial, v_pred = tf.get_default_session().run([self.action_policy_prob, self.spatial_policy_prob, self.v_preds], feed_dict={self.obs: obs})
            return action, spatial, v_pred

    def get_action_prob(self, obs):
        return tf.get_default_session().run([self.action_policy_prob, self.spatial_policy_prob], feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)