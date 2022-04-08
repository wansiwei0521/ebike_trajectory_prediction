# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-20 11:23:31
LastEditTime: 2022-04-08 00:19:41
LastEditors: Vansw
Description: train process
FilePath: //ebike_trajectory_prediction//units//train_process.py
"""
from ast import Return
from cgi import test
import sys, os
import tensorflow as tf
import gym
from tqdm import tqdm
import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import numpy as np

# self construstion
from units.loss_func import maxentirl_loss
from units.units import save_weights,load_weights,generate_single_traj,graph,graph_reward
from units.agent import Agent
from units.reward_cnn import FunctionNet



# total_timesteps = 30000
# log_interval = 30000

def train_irl_process(expert_trajs, reward_train_episode,env_id,reward_func,lr,curr_model_path,rl_paramater,env_pos_path=None,destination_func=None):

    total_timesteps,log_interval = rl_paramater
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    for i_epi in tqdm(range(reward_train_episode)):
        
        env = gym.make(env_id, reward_func=reward_func,file_path=env_pos_path,destination_func=destination_func)
        
        agent = Agent(env, total_timesteps, log_interval)
        
        test_trajs = agent.generate_agent_traj(expert_trajs)
        curr_policy_fe_trajs = env.get_feature(test_trajs)
        expert_fe_trajs = env.get_feature(expert_trajs)
        
        # print(expert_trajs[1])
        
        with tf.GradientTape() as grad:
            loss = maxentirl_loss(curr_policy_fe_trajs, expert_fe_trajs, reward_func)
            # print(loss)
        reward_func_grad = grad.gradient(loss,reward_func.trainable_weights)
        tf.optimizers.Adam(lr).apply_gradients(zip(reward_func_grad, reward_func.trainable_weights))
        
        train_loss(loss)
        
        # saving reward model
        if (i_epi % 10 == 0 and i_epi != 0) or i_epi == reward_train_episode - 1:
            
            if not os.path.isdir(curr_model_path):
                os.makedirs(curr_model_path)
            
            print(f'episode:{i_epi+1}, model saved!')
            save_weights(reward_func,curr_model_path)
            
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=i_epi)
            
        # Reset metrics every epoch
        # train_loss.reset_states()
        
    return curr_model_path

def train_rl(expert_trajs,feature_dim,hidden_dim,model_save_path,env_id,env_pos_path,curr_model_path,rl_paramater,destination_func):
    
    total_timesteps,log_interval = rl_paramater
    
    reward_func = FunctionNet(feature_dim,hidden_dim)
    reward_func.eval()
    load_weights(reward_func,model_save_path)
    
    env = gym.make(env_id, reward_func=reward_func,file_path=env_pos_path,destination_func=destination_func)
    # if env_pos_path is not None:
    #     env.set_environment_pos(env_pos_path)
    
    # check_env(env)
    model = SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    
    if not os.path.isdir(curr_model_path):
        os.makedirs(curr_model_path)
    
    model.save(curr_model_path)
    
    # expert_traj = expert_trajs[1]
    for expert_traj in expert_trajs:
        target_time = len(expert_traj)
        
        test_traj,rewards = generate_single_traj(env,model,expert_traj[1],target_time)
        
        # graph_reward(rewards)
        # print(rewards,len(test_traj))
        graph(expert_traj,test_traj)
        
        