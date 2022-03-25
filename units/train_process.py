# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-20 11:23:31
LastEditTime: 2022-03-25 14:51:19
LastEditors: Vansw
Description: train process
FilePath: //ebike_trajectory_prediction//units//train_process.py
"""
from cgi import test
import sys, os
import tensorflow as tf
import gym
from tqdm import tqdm

# self construstion
from units.loss_func import maxentirl_loss
from units.units import save_weights,load_weights,get_feature_expectation
from units.agent import Agent

total_timesteps = 30
log_interval = 30

def train_process(expert_single_traj, reward_train_episode,env_id,reward_func,lr,curr_model_path,env_pos_path=None):

    for i_epi in tqdm(range(reward_train_episode)):
        
        env = gym.make(env_id, reward_func=reward_func, target_time=100)
        if env_pos_path is not None:
            env.set_environment_pos(env_pos_path)
        
        agent = Agent(env, total_timesteps, log_interval)
        
        # 有问题 生成轨迹问题 动态起点
        test_traj = agent.generate_agent_traj(1,expert_single_traj)[0]
        
        
        curr_policy_fe_traj = get_feature_expectation(test_traj)
        expert_fe_trajs = get_feature_expectation(expert_single_traj)
        
        with tf.GradientTape() as grad:
            loss = maxentirl_loss(curr_policy_fe_traj, expert_fe_trajs, reward_func)
        reward_func_grad = grad.gradient(loss,reward_func.trainable_weights)
        tf.optimizers.Adam(lr).apply_gradients(zip(reward_func_grad, reward_func.trainable_weights))
        
        # saving reward model
        if i_epi % 100 == 0 or i_epi == reward_train_episode - 1:
            print(f'episode:{i_epi}, model saved!')
            save_weights(reward_func,curr_model_path)