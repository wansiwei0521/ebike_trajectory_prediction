# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-09 10:06:15
LastEditTime: 2022-03-29 11:06:42
LastEditors: Vansw
Description: IntersectionEnv with traffic signal
FilePath: //ebike_trajectory_prediction//env//IntersectionEnv.py
"""
from gym import spaces, core
from gym.utils import seeding
import numpy as np
import pandas as pd
import tensorflow as tf
from ruamel.yaml import YAML

class IntersectionEnv(core.Env):
    """
    Description:
        Prediction and training in target_time increments, which default value is ten-second

    Source:

    Observation:
        Type: Box(8)
        Num     Observation               Min                  Max      Feature
        0       ebike Position x          -3000               3000      
        1       ebike Position y          -3000               3000      middle or side of the road
        2       ebike Vx                  -10                 10        
        3       ebike Vy                  -10                 10        speeding
        4       nearest car dis            0                  1000       grade
        5       traffic signal             0                  20        greeb,red = 1,0
        6       time                      -Inf                Inf       
        7       waiting time               0                  Inf        waiting time

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
        
        # feature init
        self.road_x_top = None
        self.road_x_down = None
        self.road_y_top = None
        self.road_y_down = None
        self.speeding = None
        self.safe_dis = None
        self.fps_second = None
        self.feature_gamma = None
        self.wait_time_threshold = None
        
        # other init
        self.interval_time = 1 # fps
        self.crash_threshould = 1
        self.target_time = target_time
        self.reward_func = reward_func
        self.environment_car_pos = None
        self.threshold_vel = 0.3 # is moving or not 5/24
        self.time_count = None
        self.vel_limited = 6.5
        
        # dimension
        self.observation_dim = 8
        self.action_dim = 2
        
        # threshould of the obs and act space
        self.pos_threshould = 3000
        self.vel_threshould = 10
        self.car_pos_threshould_min = 0
        self.car_pos_threshould_max = 1000
        self.traffic_signal_min = 0
        self.traffic_signal_max = 20
        self.last_time_threshould_min = 0
        # self.last_time_threshould_max = 10
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
            np.finfo(np.float32).max
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
        
        # self._A = np.array([[0,0],
        #                     [0,0]])
        # self._B = np.array([[1,0],
        #                     [0,1]]) 
        
        
        # environment pos
        # self.env_pos_file_dir = None
        # self.env_pos_encoding = None
        # self.environment_car_pos = None
        # self._get_environment_car_pos()
        
        
        # self.steps_beyond_done = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def set_environment_pos(self, file_path=None, encoding="utf-8"):
        # if car_file_path:
        #     self.environment_car_pos = pd.read_csv(car_file_path, encoding=encoding)
            
        if file_path:
            yaml = YAML()
            config = yaml.load(open(file_path))
            self.road_x_top, self.road_x_down, self.road_y_top, self.road_y_down = config['road_boundary']
            self.speeding = config['speeding']
            self.safe_dis = config['safe_distence']
            self.fps_second = config['fps_second']
            self.wait_time_threshold = config['wait_time_threshold']
            self.feature_gamma = config['feature_gamma']
            car_file_path = config['car_file_path']
            self.environment_car_pos = pd.read_csv(car_file_path, encoding=encoding)
        pass
        
    def get_state_feature(self, state):
        
        x,y,vx,vy,near_car_dis,traffic_sign,time,last_time = state
        
        # middle 1 or side 0 of the intersection
        loc_of_road = 1 if self.road_x_down<x<self.road_x_top and self.road_y_down<y<self.road_y_top else 0
        
        # speeding: 1 for speeding
        vel = np.array([vx,vy])
        speeding = 1 if np.linalg.norm(vel)>self.speeding else 0
        
        # distence of car: safe for 1
        distence_of_car = 1 if near_car_dis>=self.safe_dis else 0
        
        # traffic sign: red for 1
        traf_sign = 1 if traffic_sign>0 else 0
        
        # waiting time
        wait_time_thre = self.fps_second * self.wait_time_threshold
        wait = 1 if last_time>=wait_time_thre else 0
        
        feature = [loc_of_road, speeding, distence_of_car, traf_sign, wait]
        
        return np.array(feature)
    
    def get_feature(self, traj):
        feature_traj = []
        for t in range(0,len(traj)):
            curr_obs = (self.feature_gamma**t)*self.get_state_feature(traj[t])
            feature_traj.append(curr_obs)
        return np.array(feature_traj)
    
    # transiton model
    # def f(self,x,u): 
    #     return self._A @ x + self._B @ u 
    
    def reset(self, ordinary_state=None):
        
        self.time_count = 0 # time counting
        
        if ordinary_state is None:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.observation_dim,))
        else:
            self.state = ordinary_state
        self.steps_beyond_done = None
        return np.array(self.state)
	
    def step(self, action):
        
        self.time_count += 1
        
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
        
        # ebike max velosity limitation
        vel = np.array([vx,vy])
        vel = np.clip(vel,-self.vel_limited,self.vel_limited)
        vx, vy = vel
        
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
        # last_time += self.interval_time
        # threshold_vel = 3 # 3 px/fps
        if abs(vx) <= self.threshold_vel and abs(vy) <= self.threshold_vel:
            last_time += self.interval_time
        else:
            last_time = 0
        
        self.state = (x,y,vx,vy,near_car_dis,traffic_sign,time,last_time)
        
        return np.array(self.state)

    def _get_reward(self, done):
        # ! target zone?

        if self.reward_func is not None:
            fe_tensor = tf.convert_to_tensor(np.array(self.get_state_feature(self.state)), dtype=np.float32)
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
            or self.time_count > self.target_time
            or not -self.pos_threshould<=self.state[0]<=self.pos_threshould
            or not -self.pos_threshould<=self.state[0]<=self.pos_threshould
        )
        
        return done
    
    def close(self):
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
        pass

