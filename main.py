# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 12:56:00
LastEditTime: 2022-04-08 19:50:24
LastEditors: Vansw
Description: main process
FilePath: //ebike_trajectory_prediction//main.py
"""
import sys, os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import gym
import datetime
np.set_printoptions(threshold=sys.maxsize)

# self construstion
import env
from units.reward_cnn import FunctionNet,DsFunctionNet
from units.train_process import train_irl_process, train_rl, graph
from units.destination_estimate import destination_train

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
env_pos_path = "D:\MyProject\ebike_trajectory_prediction\configs\IntersectionEnv_config.yml"
print(f"{len(expert_trajs)} trajectories loaded!")

obs_dim = 4
output_layer = 2
feature_dim = 7
hidden_dim = 64
reward_train_episode = 50
lr = 3e-4
rl_paramater = [3000,3000]

destination_func = DsFunctionNet(obs_dim,hidden_dim,output_layer)
# destination_func.train()
destination_func.eval()

# destination_train(expert_trajs,destination_func,curr_model_path)
model_path = "./trained_models/IntersectionEnv-v1/2022-04-08_15-52-35/models/"
tl.files.load_and_assign_npz(name=model_path+'/destination.npz', network=destination_func)

# reward_func = FunctionNet(obs_dim,hidden_dim)
reward_func = FunctionNet(feature_dim,hidden_dim)
reward_func.train()

expert_trajs = expert_trajs[6]
expert_trajs = np.expand_dims(expert_trajs,axis=0)

# fe_tensor = tf.convert_to_tensor(np.array(expert_trajs[0][0][0:4]), dtype=np.float32)
# print(destination_func(fe_tensor),expert_trajs[0][-1][0:2])
# expert_trajs = expert_trajs[0]
# print(np.shape(expert_trajs))
# print(len(expert_trajs))
# expert_trajs = np.array([expert_trajs,expert_trajs])
# graph(expert_trajs)

# train process
# reward_func_path = train_irl_process(expert_trajs, reward_train_episode,env_id,reward_func,lr,curr_model_path,rl_paramater,env_pos_path,destination_func)

reward_func_path = './trained_models/IntersectionEnv-v1/2022-04-08_16-54-19/models/'
curr_model_path = './trained_models/IntersectionEnv-v1/2022-04-08_16-54-19/models/'

train_rl(expert_trajs,feature_dim,hidden_dim,reward_func_path,env_id,env_pos_path,curr_model_path,rl_paramater,destination_func)

# from units.units import graph
# graph(expert_traj=expert_trajs[:,[6,2]],test_traj=expert_trajs[:,[6,3]])
