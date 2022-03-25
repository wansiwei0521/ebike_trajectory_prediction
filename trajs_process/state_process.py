# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-25 15:22:35
LastEditTime: 2022-03-25 17:29:36
LastEditors: Vansw
Description: 
FilePath: //ebike_trajectory_prediction//trajs_process//state_process.py
"""
import numpy as np
import pandas as pd


# def set_environment_pos(self, file_dir=None, encoding="utf-8"):
#         if file_dir:
#             self.environment_car_pos = pd.read_csv(file_dir, encoding=encoding)
#         pass

def near_car(x,y,intersection_car_location,time):

    try:
        temp_df = intersection_car_location[intersection_car_location['time']==time]
        temp_df = temp_df[['x','y']]
        # del temp_df['time']
        temp_array = np.array(temp_df)
        temp_array = np.square(temp_array) - np.square(np.array([x,y]))
        near_car_dis = temp_array.sum(axis=1).min()
    except Exception:
        near_car_dis = 1000
    
    return near_car_dis

def during_time(x_vel,y_vel):
    global dur_time
    dur_time = 0
    threshold_vel = 3 # 3 px/fps
    # is_wait = 1 if abs(x_vel) <= threshold_vel and abs(y_vel) <= threshold_vel else 0
    if abs(x_vel) <= threshold_vel and abs(y_vel) <= threshold_vel:
        dur_time +=1
    else:
        dur_time = 0
    return dur_time

def state_process(traj,env_file):
    """traj: x, y, time"""
    # vel
    traj_temp = np.diff(traj,axis=0)
    x_vel = traj_temp[:,0] / traj_temp[:,2]
    y_vel = traj_temp[:,1] / traj_temp[:,2]
    # x_vel = np.apply_along_axis()
    
    coor = np.c_[x_vel,y_vel]
    traj = np.delete(traj,0,axis=0)
    traj = np.c_[traj,coor]
    # np.insert(traj,values=coor,axis=0)
    
    # nearest car
    # near_car_dis = []

    intersection_car_location = pd.read_csv(env_file, encoding="utf-8")
    near_car_dis = np.apply_along_axis(lambda x:near_car(x[0],x[1],intersection_car_location,x[2]),axis=1,arr=traj)
    # near_car_dis = near_car(x,y,intersection_car_location,time)

    
    # traffic sign
    traf_sig = np.zeros(len(traj))
    
    # dur time
    dur_time = np.apply_along_axis(lambda x:during_time(x[3],x[4]),axis=1,arr=traj)
    
    # unite
    
    traj = np.c_[traj[:,[0,1]],traj[:,[3,4]],near_car_dis,traf_sig,traj[:,2],dur_time]
    
    return traj
    
if __name__ == '__main__':
    c = np.array([[1,2,6],[4,5,7],[7,8,8],[9,10,9]])
    env_pos_path = './data/111.csv'
    traj = state_process(c,env_pos_path)
    print(traj)
