# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 10:25:10
LastEditTime: 2022-04-01 23:15:28
LastEditors: Vansw
Description: loss function
FilePath: //ebike_trajectory_prediction//units//loss_func.py
"""
import tensorflow as tf
import numpy as np

def maxentirl_loss(learners, experts, reward_func):
    
    # print(len(learners),len(experts))
    trajs = np.c_[learners, experts]
    learner_rewards = []
    expert_rewards = []
    for learner, expert in trajs:
        
        
        learner_tensor = tf.convert_to_tensor(np.array(learner,dtype=np.float32), dtype=np.float32)
        expert_tensor = tf.convert_to_tensor(np.array(expert,dtype=np.float32), dtype=np.float32)
        
        learner_reward = tf.reshape(reward_func(learner_tensor),[-1])
        expert_reward = tf.reshape(reward_func(expert_tensor),[-1])
        
        learner_rewards.append(tf.reduce_mean(learner_reward))
        expert_rewards.append(tf.reduce_mean(expert_reward))
    
    # return tf.abs(tf.reduce_mean(expert_reward) - tf.reduce_mean(learner_reward))
    return tf.losses.categorical_crossentropy(expert_rewards,learner_rewards)


if __name__ == '__main__':
    from reward_cnn import RewardFunctionNet
    reward_func = RewardFunctionNet(2,4)
    reward_func.train()
    
    learner = [[[2,2],[2,2]],[[2,2],[2,2],[2,2]]]
    expert = [[[3,3],[1,1]],[[3,3],[1,1],[1,1]]]
    
    learner = np.array(learner,dtype=object)
    expert = np.array(expert,dtype=object)
    
    a = maxentirl_loss(learner,expert,reward_func)
    print(a)
    