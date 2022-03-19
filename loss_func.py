# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 10:25:10
LastEditTime: 2022-03-18 11:04:10
LastEditors: Vansw
Description: loss function
FilePath: //Preference-Planning-Deep-IRLd://MyProject//LocalGit//thesis//something done//loss_func.py
"""
import tensorflow as tf
import numpy as np

def maxentirl_loss(learner, expert, reward_func):
    learner_tensor = tf.convert_to_tensor(np.array(learner), dtype=np.float32)
    expert_tensor = tf.convert_to_tensor(np.array(expert), dtype=np.float32)
    
    learner_reward = tf.reshape(reward_func(learner_tensor),[-1])
    expert_reward = tf.reshape(reward_func(expert_tensor),[-1])
    
    # 1000 how to define
    return 1000 * (tf.reduce_mean(learner_reward) - tf.reduce_mean(expert_reward))


if __name__ == '__main__':
    from reward_cnn import RewardFunctionNet
    reward_func = RewardFunctionNet(2,4)
    reward_func.train()
    
    learner = [[2,2],[2,2]]
    expert = [[3,3],[1,1]]
    
    a = maxentirl_loss(learner,expert,reward_func)
    print(a)
    