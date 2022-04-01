# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 13:34:03
LastEditTime: 2022-04-01 22:10:21
LastEditors: Vansw
Description: sub function
FilePath: //ebike_trajectory_prediction//units//units.py
"""

import numpy as np
import tensorlayer as tl

import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib import patches
import warnings
warnings.simplefilter('ignore')

# # extract feature
# def get_feature(state):
#     state_feature = state
#     return state_feature

# def get_feature_expectation(traj,feature_gamma=1):
#     feature_traj = []
#     for t in range(0,len(traj)):
#         curr_obs = (feature_gamma**t)*traj[t]
#         feature_traj.append(curr_obs)
#     return np.array(feature_traj)

# saving and loading model
def save_weights(reward_func, path):  # save trained weights
    tl.files.save_npz(reward_func.trainable_weights, name=path+'/model_reward.npz')
    
def load_weights(reward_func, path):  # load trained weights
    tl.files.load_and_assign_npz(name=path+'/model_reward.npz', network=reward_func)
    
def graph(expert_traj,test_traj = None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(expert_traj[:, 0], expert_traj[:, 1], color='b', label='expert')
    
    if test_traj is not None:
        ax.plot(test_traj[:, 0], test_traj[:, 1], color='y', label='test')
    
    ax.grid()
    ax.set_xlabel('x, px')
    ax.set_ylabel('y, px')
    ax.legend()
    plt.show()
    
def generate_single_traj(env, model, expert_traj_state=None, target_time=None):
    obs = env.reset(ordinary_state=expert_traj_state,target_time=target_time)
    
    single_traj = []
    done = False

    t = 0
    while not done:
        single_traj.append(obs)
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        t += 1
        
    single_traj.append(obs)
    
    single_traj = np.array(single_traj)
    
    single_traj_xy = single_traj[:,[0,1]]
    
    return single_traj_xy