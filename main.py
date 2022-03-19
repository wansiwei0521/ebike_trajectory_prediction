# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 12:56:00
LastEditTime: 2022-03-19 14:11:03
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
from reward_cnn import RewardFunctionNet
from loss_func import maxentirl_loss
from units import save_weights,load_weights
from agent import Agent

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
trajs_filepath = ""
human_trajs = np.load(trajs_filepath+"/expert_trajs.npy")
expert_trajs = np.array(human_trajs).copy()

# ! feature extract
for i in range(len(expert_trajs)):
    single_traj = expert_trajs[i]

obs_dim = env.observation_space.shape[0]
hidden_dim = 32
reward_train_episode = 200
lr = 3e-4

reward_func = RewardFunctionNet(obs_dim,hidden_dim)
# 优化器！！adam
# ! agent 有待改进
for i_epi in range(reward_train_episode):
    
    agent = Agent(gym.make(env_id, reward_func=reward_func), 30000, 30000)
    
    test_traj = agent.generate_agent_traj(1)[0]  #(10,100,4)

    curr_policy_fe_traj = test_traj
    expert_fe_trajs = expert_trajs
    
    with tf.GradientTape() as grad:
        loss = maxentirl_loss(curr_policy_fe_traj, expert_fe_trajs, reward_func)
    reward_func_grad = grad.gradient(loss,reward_func.trainable_weights)
    tf.optimizers.Adam(lr).apply_gradients(zip(reward_func_grad, reward_func.trainable_weights))
    
    # saving reward model
    if i_epi % 100 == 0 or i_epi == reward_train_episode - 1:
        save_weights(reward_func,curr_model_path)