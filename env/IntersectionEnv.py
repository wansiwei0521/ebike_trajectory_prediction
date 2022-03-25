# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-09 10:06:15
LastEditTime: 2022-03-25 17:15:58
LastEditors: Vansw
Description: IntersectionEnv with traffic signal
FilePath: //ebike_trajectory_prediction//env//IntersectionEnv.py
"""
from gym import spaces, core
from gym.utils import seeding
import numpy as np
import pandas as pd
import tensorflow as tf

class IntersectionEnv(core.Env):
    """
    Description:
        Prediction and training in target_time increments, which default value is ten-second

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

    
    def __init__(self, reward_func=None, target_time=10):
        
        super(IntersectionEnv, self).__init__()
        
        # other init
        self.interval_time = 1
        self.crash_threshould = 1
        self.target_time = target_time
        self.reward_func = reward_func
        self.environment_car_pos = None
        
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
        
    def reset(self, ordinary_state=None):
        # ordinary position do not fix
        if ordinary_state is None:
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
        # record the information of training
        info = {}
        return obs, reward, done, info

    def render(self):
        
        print("visualizing failure!")
        
        pass
    
    def _get_observation(self, action):
        
        x,y,vx,vy,near_car_dis,traffic_sign,time,last_time = self.state
        
        
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
        
        try:
            intersection_car_location = self.environment_car_pos
            temp_df = intersection_car_location[intersection_car_location['time']==time]
            temp_df = temp_df[['x','y']]
            # del temp_df['time']
            temp_array = np.array(temp_df)
            temp_array = np.square(temp_array) - np.square(np.array([x,y]))
            near_car_dis = temp_array.sum(axis=1).min()
        except Exception:
            near_car_dis = 200
            
        # traffic
        traffic_sign -= self.interval_time
        traffic_sign = 0 if traffic_sign <= 0 else traffic_sign
        
        # last time
        last_time += self.interval_time
        
        self.state = (x,y,vx,vy,near_car_dis,traffic_sign,time,last_time)
        
        return np.array(self.state)

    def _get_reward(self, done):
        # ! done or other??

        if self.reward_func is not None:
            fe_tensor = tf.convert_to_tensor(np.array(self.state), dtype=np.float32)
            reward = self.reward_func(fe_tensor).numpy().flatten()
            reward = reward if len(reward)>1 else reward[0] # unnormal get value of tensor
        else:
            reward = 0
        
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

