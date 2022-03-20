# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 12:56:00
LastEditTime: 2022-03-20 16:24:31
LastEditors: Vansw
Description: main process
FilePath: //Preference-Planning-Deep-IRLd://MyProject//ebike_trajectory_prediction//main.py
"""
import sys, os
import numpy as np
import tensorflow as tf
import gym
import datetime

# self construstion
from units.reward_cnn import RewardFunctionNet
from units.train_process import train_process

# redirect working dir
# working_dir = ""
# os.chdir(working_dir)
# print(os.getcwd())
work_dir = os.getcwd()

env_id = 'IntersectionEnv-v1'
env = gym.make(env_id,reward_func=None)


curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
curr_run_path = "trained_models/{}/{}".format(env_id, curr_time)
if not os.path.isdir(curr_run_path):
   os.makedirs(curr_run_path)

curr_model_path = curr_run_path + "/models"
if not os.path.isdir(curr_model_path):
   os.makedirs(curr_model_path)

# curr_visual_path = curr_run_path + "/visuals"
# if not os.path.isdir(curr_visual_path):
#    os.makedirs(curr_visual_path)

# loading expert trajs
trajs_filepath = work_dir+"/expert_trajs"
human_trajs = np.load(trajs_filepath+"/expert_trajs.npy")
expert_trajs = np.array(human_trajs).copy()

obs_dim = env.observation_space.shape[0]
hidden_dim = 32
reward_train_episode = 200
lr = 3e-4


reward_func = RewardFunctionNet(obs_dim,hidden_dim)
reward_func.train()

# train process
# for i in range(len(expert_trajs)):
#     expert_single_traj = expert_trajs[i]
#     train_process(expert_trajs, reward_train_episode,env_id,reward_func,lr,curr_model_path)
train_process(expert_trajs, reward_train_episode,env_id,reward_func,lr,curr_model_path)