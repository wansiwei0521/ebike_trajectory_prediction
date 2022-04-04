# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 12:56:00
LastEditTime: 2022-04-03 13:25:25
LastEditors: Vansw
Description: main process
FilePath: //ebike_trajectory_prediction//main.py
"""
import sys, os
import numpy as np
import tensorflow as tf
import gym
import datetime
np.set_printoptions(threshold=sys.maxsize)

# self construstion
import env
from units.reward_cnn import RewardFunctionNet
from units.train_process import train_irl_process, train_rl, graph

# redirect working dir
work_dir = os.getcwd()

env_id = 'IntersectionEnv-v1'
# env = gym.make(env_id,reward_func=None)

# model saved path
curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
curr_run_path = "trained_models/{}/{}".format(env_id, curr_time)
curr_model_path = curr_run_path + "/models"

# loading expert trajs
trajs_filepath = work_dir+"/data/ebike_traj"
human_trajs = np.load(trajs_filepath+"/ebike_traj.npy",allow_pickle=True)
expert_trajs = np.array(human_trajs,dtype=object).copy()
env_pos_path = './configs/IntersectionEnv_config.yml'
print(f"{len(expert_trajs)} trajectories loaded!")

# obs_dim = env.observation_space.shape[0]
feature_dim = 5
hidden_dim = 32
reward_train_episode = 400
lr = 3e-4

# reward_func = RewardFunctionNet(obs_dim,hidden_dim)
reward_func = RewardFunctionNet(feature_dim,hidden_dim)
reward_func.train()

expert_trajs = expert_trajs[6:8]
# expert_trajs = np.expand_dims(expert_trajs,axis=0)
# expert_trajs = np.array([expert_trajs,expert_trajs])
# graph(expert_trajs)

# train process
reward_func_path = train_irl_process(expert_trajs, reward_train_episode,env_id,reward_func,lr,curr_model_path,env_pos_path)

# reward_func_path = './trained_models/IntersectionEnv-v1/2022-04-03_12-35-30/models'
# curr_model_path = './trained_models/IntersectionEnv-v1//2022-04-03_12-35-30/models'

train_rl(expert_trajs,feature_dim,hidden_dim,reward_func_path,env_id,env_pos_path,curr_model_path)

# from units.units import graph
# graph(expert_traj=expert_trajs[:,[6,2]],test_traj=expert_trajs[:,[6,3]])