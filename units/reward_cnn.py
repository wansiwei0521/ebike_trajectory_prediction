# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-17 14:47:21
LastEditTime: 2022-03-20 10:38:16
LastEditors: Vansw
Description: reward function net
FilePath: //Preference-Planning-Deep-IRLd://MyProject//ebike_trajectory_prediction//reward_cnn.py
"""
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense,Input
from tensorlayer.models import Model

class RewardFunctionNet(Model):
    def __init__(self, num_inputs, hidden_dim, init_w=3e-3):
        super(RewardFunctionNet, self).__init__()
        # init weight
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        # three dense layer
        # self.inputlayer = Input(shape=[num_inputs,],dtype=tf.float32)
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='f1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='f2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='f3')

    def forward(self, input):
        # x = self.inputlayer(input)
        # x = self.linear1(x)
        input = tf.expand_dims(input, 0)
        
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
    def normalize_state(self, obs_high, obs_low, state_sample):
        space = obs_high - obs_low
        norm_sample = state_sample/space
        return norm_sample
    
if __name__ == '__main__':
    # test
    import gym
    id = "IntersectionEnv-v1"
    env = gym.make(id)
    state = env.reset()
    print(state)
    fe_tensor = tf.convert_to_tensor(np.array(state), dtype=np.float32)
    print(fe_tensor)
    
    # fe = tf.expand_dims(fe_tensor, 0)
    reward = RewardFunctionNet(8,32)
    reward.train()
    print(reward)
    r = reward(fe_tensor).numpy().flatten()[0]
    
    print(r)