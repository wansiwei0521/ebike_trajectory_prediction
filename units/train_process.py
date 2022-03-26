# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-20 11:23:31
LastEditTime: 2022-03-26 20:37:27
LastEditors: Vansw
Description: train process
FilePath: //ebike_trajectory_prediction//units//train_process.py
"""
from cgi import test
import sys, os
import tensorflow as tf
import gym
from tqdm import tqdm
import datetime

# self construstion
from units.loss_func import maxentirl_loss
from units.units import save_weights,load_weights
from units.agent import Agent

total_timesteps = 200
log_interval = 200

def train_process(expert_single_traj, reward_train_episode,env_id,reward_func,lr,curr_model_path,env_pos_path=None):

    target_time = len(expert_single_traj)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    for i_epi in tqdm(range(reward_train_episode)):
        
        env = gym.make(env_id, reward_func=reward_func, target_time=target_time)
        if env_pos_path is not None:
            env.set_environment_pos(env_pos_path)
        
        agent = Agent(env, total_timesteps, log_interval)
        
        # 有问题 生成轨迹问题 动态起点
        test_traj = agent.generate_agent_traj(1, expert_single_traj)[0]
        
        
        curr_policy_fe_traj = env.get_feature(test_traj)
        expert_fe_trajs = env.get_feature(expert_single_traj)
        
        with tf.GradientTape() as grad:
            loss = maxentirl_loss(curr_policy_fe_traj, expert_fe_trajs, reward_func)
        reward_func_grad = grad.gradient(loss,reward_func.trainable_weights)
        tf.optimizers.Adam(lr).apply_gradients(zip(reward_func_grad, reward_func.trainable_weights))
        
        train_loss(loss)
        
        # saving reward model
        if i_epi % 100 == 0 or i_epi == reward_train_episode - 1:
            print(f'episode:{i_epi+1}, model saved!')
            save_weights(reward_func,curr_model_path)
            
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=i_epi)
            
        # Reset metrics every epoch
        # train_loss.reset_states()