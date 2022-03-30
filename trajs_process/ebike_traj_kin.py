# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-24 17:16:43
LastEditTime: 2022-03-29 10:02:03
LastEditors: Vansw
Description: 
FilePath: //ebike_trajectory_prediction//trajs_process//ebike_traj_kin.py
"""
import numpy as np
import csv
import os
from traj_smooth import smooth_dataset
from trajs_process.state_process import state_process
from units.units import graph

filename='./trajs_process/电动车轨迹.CSV'
restore_sig = 0
# data = []
# trajs = {}
curr_path = os.getcwd()
traj_names = []
trajs_num = 0
smoothing_window = 11
env_pos_path = './data/111.csv'

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    #header = next(csv_reader)
    for row in csv_reader:
        # if row[0]
        if row[0] == '':
            restore_sig = 0
        
        if row[0]=='Label :':
            if trajs_num != 0:
                traj = np.array(traj)
                
                save_path = curr_path+f"/data/trajs/{trajs_num}_{traj_name}.npy"
                
                # change type
                traj = traj.astype(np.float32)
                # traj = np.abs(traj)
                traj[:,1] = traj[:,1] + np.array([1512])
                traj_smooth =  smooth_dataset(smoothing_window, traj)
                
                # state process
                traj_state_process = state_process(traj,env_pos_path)
                
                np.save(save_path,traj_state_process)
                print(save_path)
                
                graph(traj_state_process)
                
            traj = []
            traj_name = row[1]
            traj_names.append(traj_name)
            trajs_num += 1
            
        if restore_sig:
            
            traj.append(row)

        if row[0] == 'x':
            restore_sig = 1
        
        # data.append(row[5])
        # print(type(float(row[0])))
    # print(trajs)
    

