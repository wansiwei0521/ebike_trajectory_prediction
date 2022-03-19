# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-09 10:06:15
LastEditTime: 2022-03-18 14:35:38
LastEditors: Vansw
Description: IntersectionEnv without traffic signal
FilePath: //Preference-Planning-Deep-IRLd://MyProject//LocalGit//thesis//something done//IntersectionEnv.py
"""
from gym import spaces, core
from gym.utils import seeding
import numpy as np
import pandas as pd
# import torch
import tensorflow as tf

# core.Env 是 gym 的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
# 必须要重写的方法有: 
# __init__()：构造函数
# reset()：初始化环境
# step()：环境动作,即环境对agent的反馈
# render()：如果要进行可视化则实现


# class Car():
#     def __init__(self, trac):
#         self.trac = trac
#         pass


class IntersectionEnv(core.Env):
    """
    Description:
        十秒为单位进行预测、训练

    Source:

    Observation:
        Type: Box(8)
        Num     Observation               Min                     Max
        0       ebike Position x          -1000                  1000
        1       ebike Position y          -1000                  1000
        2       ebike Vx                  -20                    20
        3       ebike Vy                  -20                    20
        4       nearest car dis            0                     100
        5       traffic signal             0                     20
        6       time                      -Inf                   Inf
        7       duration(time last)        0                     10

    Actions:
        Type: Box(2)
        Num     Action                    Min                     Max
        0       Ax                        -3                      3
        1       Ay                        -3                      3

    Reward:
        

    Starting State:
        

    Episode Termination:
        
    """

    
    def __init__(self, reward_func=None):
        
        super(IntersectionEnv, self).__init__()
        
        # other init
        self.interval_time = 1
        self.crash_threshould = 1
        self.target_time = 10
        self.reward_func = reward_func
        
        # dimension
        self.observation_dim = 8
        self.action_dim = 2
        
        # threshould of the obs and act space
        self.pos_threshould = 1000
        self.vel_threshould = 20
        self.car_pos_threshould_min = 0
        self.car_pos_threshould_max = 100
        self.traffic_signal_min = 0
        self.traffic_signal_max = 20
        self.last_time_threshould_min = 0
        self.last_time_threshould_max = 10
        self.acc_threshould = 3
        
        act_high = np.array([
            self.acc_threshould,
            self.acc_threshould
        ],dtype=np.float32)
        
        obs_high = np.array([
            self.pos_threshould,
            self.pos_threshould,
            self.vel_threshould,
            self.vel_threshould,
            self.car_pos_threshould_max,
            self.traffic_signal_max,
            np.finfo(np.float32).max,
            self.last_time_threshould_max
        ],dtype=np.float32)
        
        obs_low = np.array([
            -self.pos_threshould,
            -self.pos_threshould,
            -self.vel_threshould,
            -self.vel_threshould,
            self.car_pos_threshould_min,
            self.traffic_signal_min,
            -np.finfo(np.float32).max,
            self.last_time_threshould_min
        ],dtype=np.float32)
        
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space =  spaces.Box(obs_low,obs_high,dtype=np.float32)
        
        
        # visualize
        # self.viewer = None
        self.seed()
        self.state = None
        
        # environment pos
        # self.env_pos_file_dir = None
        # self.env_pos_encoding = None
        # self.environment_car_pos = None
        # self._get_environment_car_pos()
        
        
        # self.steps_beyond_done = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def set_environment_pos(self, file_dir=None, encoding="utf-8"):
        if file_dir:
            self.environment_car_pos = pd.read_csv(file_dir, encoding=encoding)
        pass
        
    def reset(self,random_state=True,ordinary_state =None):
        # ordinary position do not fix
        if random_state:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.observation_dim,))
        else:
            self.state = ordinary_state
        self.steps_beyond_done = None
        return np.array(self.state)
	
    def step(self, action):
        # such a important thing!!
        obs = self._get_observation(action)
        done = self._get_done()
        reward = self._get_reward(done)
        # 用于记录训练过程中的环境信息,便于观察训练状态
        info = {}
        return obs, reward, done, info

    def render(self):
        
        print("visualizing failure!")
        
        pass
    
    def _get_observation(self, action):
        
        x,y,vx,vy,near_car_dis,traffic_sign,time,last_time = self.state
        intersection_car_location = self.environment_car_pos
        
        # Kinesiology
        x = vx * self.interval_time + \
            0.5 * action[0] * self.interval_time **2
        y = vy * self.interval_time + \
            0.5 * action[1] * self.interval_time **2
        vx = vx + action[0] * self.interval_time
        vy = vy + action[1] * self.interval_time
        
        time += self.interval_time
        
        # nearest car
        # ! 轨迹只保留碰撞之前
        temp_df = intersection_car_location[intersection_car_location['time']==time]
        del temp_df['time']
        temp_array = np.array(temp_df)
        temp_array = np.square(temp_array) - np.square(np.array([x,y]))
        near_car_dis = temp_array.sum(axis=1).min()
        
        # traffic
        traffic_sign -= self.interval_time
        traffic_sign = 0 if traffic_sign <= 0 else traffic_sign
        
        # last time
        last_time += self.interval_time
        
        self.state = (x,y,vx,vy,near_car_dis,traffic_sign,time,last_time)
        
        return np.array(self.state)

    def _get_reward(self, done):
        # ! random reward function
        # ! done or other??
        
        reward = 0
        
        if self.reward_func is not None:
            fe_tensor = tf.convert_to_tensor(np.array(self.state), dtype=np.float32)
            reward = self.reward_func(fe_tensor)
        
        # feature_num = 8 
        # theta = np.random.normal(0, 1, size=feature_num)
        # reward = np.dot(theta,self.state)
        
        return reward

    def _get_done(self):
        
        done = bool(
            self.state[4] <= self.crash_threshould
            or self.state[7] > self.target_time
            or not -self.pos_threshould<=self.state[0]<=self.pos_threshould
            or not -self.pos_threshould<=self.state[0]<=self.pos_threshould
        )
        
        return done
    
    def close(self):
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
        pass

