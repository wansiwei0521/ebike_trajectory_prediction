# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-19 10:06:34
LastEditTime: 2022-04-01 22:31:58
LastEditors: Vansw
Description: generate training trajs
FilePath: //ebike_trajectory_prediction//units//agent.py
"""

from stable_baselines3 import SAC, PPO, DDPG, DQN, HerReplayBuffer
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

import numpy as np

class Agent:
    def __init__(self, env, total_timesteps, log_interval):
        # self.env = DummyVecEnv([lambda : env])
        self.env = env
        # check_env(self.env)
        self.model = SAC('MlpPolicy', self.env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

    def generate_agent_traj(self, expert_trajs=None):
        
        trajs = []
        
        if expert_trajs is not None:
            for expert_traj in expert_trajs:
            
                # for expert_traj_state in expert_trajs[i]:
                #     trajs.append(self._generate_single_trajs(expert_traj_state))
                
                trajs.append(self._generate_single_trajs(expert_traj[0],len(expert_traj)))
                
        else:
                trajs.append(self._generate_single_trajs())

        return np.array(trajs)
    
    def _generate_single_trajs(self, expert_traj_state=None,target_time=None):
        obs = self.env.reset(ordinary_state=expert_traj_state, target_time=target_time)
        
        single_traj = []
        done = False

        t = 0
        while not done:
            single_traj.append(obs)
            action, _states = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.env.step(action)

            t += 1
            
        single_traj.append(obs)
        return single_traj
        

