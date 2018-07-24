import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from file_writer import open_file_and_save
from a2c import a2c, disconut_rewards

FLAGS = flags.FLAGS
FLAGS(sys.argv)

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP       = actions.FUNCTIONS.no_op.id
_NOT_QUEUED  = [0]
_QUEUED = [1]
_SELECT_ALL  = [0]

env = sc2_env.SC2Env(map_name='MoveToBeacon',
                    agent_interface_format=sc2_env.parse_agent_interface_format(
                        feature_screen=16,
                        feature_minimap=16,
                        rgb_screen=None,
                        rgb_minimap=None,
                        action_space=None,
                        use_feature_units=False),
                    step_mul=4,
                    game_steps_per_episode=None,
                    disable_fog=False,
                    visualize=False)
with tf.Session() as sess:
    A2C = a2c(sess, 0.00001)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess, "PositionBeacon/tmp/model.ckpt")

    for episodes in range(100000):
        states = np.empty(shape=[0, 16*16*2])
        actions_list = np.empty(shape=[0, 3])
        spatial_list = np.empty(shape=[0, 16*16])
        next_states = np.empty(shape=[0, 16*16*2])
        rewards = np.empty(shape=[0, 1])

        obs = env.reset()

        done = False
        global_step = 0
        
        marine_map = (obs[0].observation.feature_screen.base[5] == 1)
        beacon_map = (obs[0].observation.feature_screen.base[5] == 3)
        state = np.dstack([marine_map, beacon_map]).reshape(16*16*2).astype(int)

        while not done:
            global_step += 1
            action_policy, spatial_policy = A2C.choose_action(state)

            available_action = obs[0].observation.available_actions
            y, z, k = np.zeros(3), np.zeros(3), 0
            if 331 in available_action: y[0] = 1        # move screen
            if 7 in available_action: y[1] = 1          # select army
            if 0 in available_action: y[2] = 1          # no_op
            for i, j in zip(y, action_policy[0]):       # masking action
                z[k] = i*j
                k += 1

            #print(z)
            #print(spatial_policy)
            action = np.random.choice(3, p=z/sum(z))           # sampling action
            position = np.random.choice(16*16, p=spatial_policy[0])

            x, y = int(position % 16), int(position//16)    # get x, y
            
            if action == 0: actions_ = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [x, y]])
            if action == 1: actions_ = actions.FunctionCall(actions.FUNCTIONS.select_army.id, [_SELECT_ALL])
            if action == 2: actions_ = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

            # practical action = action, state = state, spatial_position = spatial_policy
            
            obs = env.step(actions=[actions_])
            marine_map = (obs[0].observation.feature_screen.base[5] == 1)
            beacon_map = (obs[0].observation.feature_screen.base[5] == 3)
            next_state = np.dstack([marine_map, beacon_map]).reshape(16*16*2).astype(int)
            reward = obs[0].reward
            done = obs[0].step_type == environment.StepType.LAST

            one_hot_action = np.zeros(3)
            one_hot_spatial = np.zeros(16*16)
            one_hot_action[action] = 1
            one_hot_spatial[position] = 1
            
            states = np.vstack([states, state])
            next_states = np.vstack([next_states, next_state])
            rewards = np.vstack([rewards, reward])
            actions_list = np.vstack([actions_list, one_hot_action])
            spatial_list = np.vstack([spatial_list, one_hot_spatial])

            if done:
                states = states.astype(float)
                next_states = next_states.astype(float)
                rewards = rewards.astype(float)
                actions_list = actions_list.astype(float)
                spatial_list = spatial_list.astype(float)
                discounted_rewards = disconut_rewards(rewards)
                A2C.learn(states, next_states, discounted_rewards, actions_list, spatial_list)
                saver.save(sess, "PositionBeacon/tmp/model.ckpt")
                print(episodes, sum(rewards))
                open_file_and_save('PositionBeacon/reward.csv', [sum(rewards)])

            state = next_state


env.close()