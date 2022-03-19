# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 13:34:03
LastEditTime: 2022-03-19 14:10:26
LastEditors: Vansw
Description: sub function
FilePath: //Preference-Planning-Deep-IRLd://MyProject//ebike_trajectory_prediction//units.py
"""

import numpy as np
import tensorlayer as tl

# extract feature
def get_feature(state):
    state_feature = state
    return state_feature

def get_feature_expectation(traj,feature_gamma=1):
    feature_traj = []
    for t in range(0,len(traj)):
        curr_obs = (feature_gamma**t)*traj[t]
        feature_traj.append(curr_obs)
    return np.array(feature_traj)

# saving and loading model
def save_weights(reward_func, path):  # save trained weights
    tl.files.save_npz(reward_func.trainable_weights, name=path+'/model_q_net1.npz')
    
def load_weights(reward_func, path):  # load trained weights
    tl.files.load_and_assign_npz(name=path+'/model_q_net1.npz', network=reward_func)