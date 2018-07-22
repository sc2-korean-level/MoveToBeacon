import tensorflow as tf
import copy


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.position = tf.placeholder(dtype=tf.int32, shape=[None], name='position')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

            action_policy_probs = self.Policy.action_policy_prob
            spatial_policy_probs = self.Policy.spatial_policy_prob
            
            action_policy = action_policy_probs * tf.one_hot(indices=self.actions, depth=action_policy_probs.shape[1])
            spatial_policy = spatial_policy_probs * tf.one_hot(indices=self.position, depth=spatial_policy_probs.shape[1])
            
            act_probs_action = tf.reduce_sum(action_policy, axis=1)
            act_probs_spatial = tf.reduce_sum(spatial_policy, axis=1)
            act_probs = act_probs_action * act_probs_spatial

            action_policy_probs_old = self.Old_Policy.action_policy_prob
            spatial_policy_probs_old = self.Old_Policy.spatial_policy_prob
                                                        
            action_policy_probs_old = action_policy_probs_old * tf.one_hot(indices=self.actions, depth=action_policy_probs_old.shape[1])
            spatial_policy_probs_old = spatial_policy_probs_old * tf.one_hot(indices=self.position, depth=spatial_policy_probs_old.shape[1])
            act_probs_action_old = tf.reduce_sum(action_policy_probs_old, axis=1)
            act_probs_spatial_old = tf.reduce_sum(spatial_policy_probs_old)
            act_probs_old = act_probs_action_old * act_probs_spatial_old

            with tf.variable_scope('loss/clip'):
                # ratios = tf.divide(act_probs, act_probs_old)
                ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                loss_clip = tf.reduce_mean(loss_clip)
                tf.summary.scalar('loss_clip', loss_clip)

            # construct computation graph for loss of value function
            with tf.variable_scope('loss/vf'):
                v_preds = self.Policy.v_preds
                loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
                loss_vf = tf.reduce_mean(loss_vf)
                tf.summary.scalar('loss_vf', loss_vf)

            # construct computation graph for loss of entropy bonus
            #with tf.variable_scope('loss/entropy'):
            #    entropy = -tf.reduce_sum(self.Policy.act_probs *
            #                            tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            #    entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
            #    tf.summary.scalar('entropy', entropy)

            with tf.variable_scope('loss'):
                loss = loss_clip - c_1 * loss_vf #+ c_2 * entropy
                loss = -loss  # minimize -loss == maximize loss
                tf.summary.scalar('loss', loss)

            self.merged = tf.summary.merge_all()
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def check(self, obs, actions, spatial, rewards, v_preds_next, gaes):
        print(actions, spatial, rewards, v_preds_next, gaes)
        result = [self.action_policy_probs,
            self.action_policy,
            self.act_probs_action,
            self.spatial_policy_probs,
            self.spatial_policy,
            self.act_probs_spatial,
            self.act_probs]
        a = tf.get_default_session().run(result, feed_dict={self.Policy.obs: obs,
                                                                self.Old_Policy.obs: obs,
                                                                self.actions: actions,
                                                                self.position: spatial,
                                                                self.rewards: rewards,
                                                                self.v_preds_next: v_preds_next,
                                                                self.gaes: gaes})
        print(a)

    def train(self, obs, actions, spatial, rewards, v_preds_next, gaes):
        tf.get_default_session().run([self.train_op], feed_dict={self.Policy.obs: obs,
                                                                self.Old_Policy.obs: obs,
                                                                self.actions: actions,
                                                                self.position: spatial,
                                                                self.rewards: rewards,
                                                                self.v_preds_next: v_preds_next,
                                                                self.gaes: gaes})
    
    def assign_policy_parameters(self):
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes