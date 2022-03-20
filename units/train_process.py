# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-20 11:23:31
LastEditTime: 2022-03-20 11:30:22
LastEditors: Vansw
Description: train process
FilePath: //Preference-Planning-Deep-IRLd://MyProject//ebike_trajectory_prediction//units//train_process.py
"""
import sys, os
import tensorflow as tf
import gym

# self construstion
from units.loss_func import maxentirl_loss
from units.units import save_weights,load_weights,get_feature_expectation
from units.agent import Agent

total_timesteps = 30000
log_interval = 30000

def train_process(expert_single_traj, reward_train_episode,env_id,reward_func,lr,curr_model_path):

    for i_epi in range(reward_train_episode):
        
        agent = Agent(gym.make(env_id, reward_func=reward_func), total_timesteps, log_interval)
        
        test_traj = agent.generate_agent_traj(1,expert_single_traj)[0]

        curr_policy_fe_traj = get_feature_expectation(test_traj)
        expert_fe_trajs = get_feature_expectation(expert_single_traj)
        
        with tf.GradientTape() as grad:
            loss = maxentirl_loss(curr_policy_fe_traj, expert_fe_trajs, reward_func)
        reward_func_grad = grad.gradient(loss,reward_func.trainable_weights)
        tf.optimizers.Adam(lr).apply_gradients(zip(reward_func_grad, reward_func.trainable_weights))
        
        # saving reward model
        if i_epi % 100 == 0 or i_epi == reward_train_episode - 1:
            save_weights(reward_func,curr_model_path)