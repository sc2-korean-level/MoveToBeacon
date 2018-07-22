import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from policy_net import Policy_net
from ppo import PPOTrain
from file_writer import open_file_and_save

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
                    visualize=True)
with tf.Session() as sess:
    Policy = Policy_net('policy')
    Old_Policy = Policy_net('old_policy')
    PPO = PPOTrain(Policy, Old_Policy, gamma=0.95)
    #sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "PositionBeacon/tmp/model.ckpt")

    for episodes in range(100000):
        observations = []
        actions_list = []
        spatial_position = []
        v_preds = []
        rewards = []

        obs = env.reset()

        done = False
        global_step = 0
        
        marine_map = (obs[0].observation.feature_screen.base[5] == 1)
        beacon_map = (obs[0].observation.feature_screen.base[5] == 3)
        state = np.dstack([marine_map, beacon_map]).reshape(16*16*2).astype(int)

        while not done:
            global_step += 1
            action_policy, spatial_policy, v_pred = Policy.act(obs=[state], stochastic=True)
            spatial_policy = np.random.choice(16*16, p=spatial_policy[0])
            v_pred = np.asscalar(v_pred)
            
            available_action = obs[0].observation.available_actions
            y, z, k = np.zeros(3), np.zeros(3), 0
            if 331 in available_action: y[0] = 1        # move screen
            if 7 in available_action: y[1] = 1          # select army
            if 0 in available_action: y[2] = 1          # no_op
            for i, j in zip(y, action_policy[0]):       # masking action
                z[k] = i*j
                k += 1
            action = np.random.choice(3, p=z/sum(z))           # sampling action
            x, y = int(spatial_policy % 16), int(spatial_policy//16)    # get x, y
            
            if action == 0: actions_ = actions.FunctionCall(_MOVE_SCREEN, [_QUEUED, [x, y]])
            if action == 1: actions_ = actions.FunctionCall(actions.FUNCTIONS.select_army.id, [_SELECT_ALL])
            if action == 2: actions_ = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

            # practical action = action, state = state, spatial_position = spatial_policy
            
            obs = env.step(actions=[actions_])
            marine_map = (obs[0].observation.feature_screen.base[5] == 1)
            beacon_map = (obs[0].observation.feature_screen.base[5] == 3)
            next_state = np.dstack([marine_map, beacon_map]).reshape(16*16*2).astype(int)
            reward = obs[0].reward
            #if reward == 0:
            #    reward = -0.1
            done = obs[0].step_type == environment.StepType.LAST
            
            observations.append(state)
            actions_list.append(action)
            spatial_position.append(spatial_policy)
            v_preds.append(v_pred)
            rewards.append(reward)
            
            if done:
                v_preds_next = v_preds[1:] + [0]
                gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)
                observations = np.reshape(observations, newshape=[-1, 16*16*2])
                actions_list = np.array(actions_list).astype(dtype=np.int32)
                spatial_position = np.array(spatial_position).astype(dtype=np.int32)
                rewards = np.array(rewards).astype(dtype=np.float32)
                v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
                gaes = np.array(gaes).astype(dtype=np.float32)
                PPO.assign_policy_parameters()
                inp = [observations, actions_list, spatial_position, rewards, v_preds_next, gaes]

                for epoch in range(5):
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                    
                    PPO.train(obs=sampled_inp[0],
                                actions=sampled_inp[1],
                                spatial=sampled_inp[2],
                                rewards=sampled_inp[3],
                                v_preds_next=sampled_inp[4],
                                gaes=sampled_inp[5])
                
                """
                for epoch in range(1):
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=1)  # indices are in [low, high)
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                    
                    PPO.check(obs=sampled_inp[0],
                                actions=sampled_inp[1],
                                spatial=sampled_inp[2],
                                rewards=sampled_inp[3],
                                v_preds_next=sampled_inp[4],
                                gaes=sampled_inp[5])
                """
                open_file_and_save('PositionBeacon/reward.csv', [sum(rewards)])
                saver.save(sess, "PositionBeacon/tmp/model.ckpt")
                print(episodes, sum(rewards))
            
            state = next_state
env.close()