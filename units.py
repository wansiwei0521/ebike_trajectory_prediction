# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 13:34:03
LastEditTime: 2022-03-18 13:38:28
LastEditors: Vansw
Description: sub function
FilePath: //Preference-Planning-Deep-IRLd://MyProject//LocalGit//thesis//something done//units.py
"""

import numpy as np

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