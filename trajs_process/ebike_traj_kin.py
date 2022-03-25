# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-24 17:16:43
LastEditTime: 2022-03-25 14:16:35
LastEditors: Vansw
Description: 
FilePath: //ebike_trajectory_predictiond://MyProject//LocalGit//thesis//trajs_process//ebike_traj_kin.py
"""
import numpy as np
import csv
import os
from traj_smooth import smooth_dataset

filename='./电动车轨迹.CSV'
restore_sig = 0
# data = []
# trajs = {}
curr_path = os.getcwd()
traj_names = []
trajs_num = 0
smoothing_window = 11

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
                # trajs['traj_name'] = traj
                # print(traj)
                # if traj_name in traj_names:
                #     save_path = curr_path+f"\{traj_name}{trajs_num}.npy"
                # else:
                #     save_path = curr_path+f"\{traj_name}.npy"
                save_path = curr_path+f"\{trajs_num}_{traj_name}.npy"
                print(save_path)
                
                # change type
                traj = traj.astype(np.float32)
                traj_smooth =  smooth_dataset(smoothing_window, traj)
                
                np.save(save_path,traj_smooth)
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
    

