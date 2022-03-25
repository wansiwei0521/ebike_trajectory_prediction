# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-25 10:46:01
LastEditTime: 2022-03-25 14:07:21
LastEditors: Vansw
Description: 
FilePath: //ebike_trajectory_predictiond://MyProject//LocalGit//thesis//trajs_process//traj_smooth.py
"""
from scipy import signal
import pandas as pd
import numpy as np
import numexpr
import os
from kf import KalmanFilterConstCoefficient

import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib import patches
import warnings
warnings.simplefilter('ignore')

# global time_column, local_x, local_y, v_vel, v_acc
time_column, local_x, local_y, v_vel, v_acc  = 2, 0, 1, 3, 4

def get_kf_x_y(dataset, alpha=0.32):
    kf = KalmanFilterConstCoefficient(alpha)
    init_dataset = kf.filtred_coordinates(dataset[:, local_x], dataset[:, local_y])
    return init_dataset

def get_smoothed_x_y(dataset, window):
    
    init_dataset = get_kf_x_y(dataset)
    
    smoothed_x_values = signal.savgol_filter(init_dataset[:, 0], window, 3)
    smoothed_y_values = signal.savgol_filter(init_dataset[:, 1], window, 3)
    # print(smoothed_x_values)

    return smoothed_x_values, smoothed_y_values

def get_smoothed_vel_accel(smoothed_x_values, smoothed_y_values, time_values, initial_vel, initial_accel):
    #create matrix of A containing current x and y and matrix B containing next x and y values
    x_y_matrix_A = np.column_stack((smoothed_x_values, smoothed_y_values))
    x_y_matrix_B = x_y_matrix_A [1:, :]
    #remove last row as it has no next values
    x_y_matrix_A = x_y_matrix_A[0:-1, :]

    # compute distance travelled between current and next x, y values
    dist_temp = numexpr.evaluate('sum((x_y_matrix_B - x_y_matrix_A)**2, 1)')
    dist = numexpr.evaluate('sqrt(dist_temp)')

    # create matrix A containing current time values, and matrix B containing next time values
    t_matrix_A = time_values
    t_matrix_B = t_matrix_A [1:]
    # remove last row
    t_matrix_A = t_matrix_A[0:-1]

    # evaluate smoothed velocity by dividing distance over delta time
    vel = numexpr.evaluate('dist * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_velocities = np.insert(vel, 0, initial_vel, axis=0)

    # create matrix A containing current velocities and matrix B containing next velocities
    vel_matrix_A = smoothed_velocities
    vel_matrix_B = vel_matrix_A [1:]
    # remove last row
    vel_matrix_A = vel_matrix_A[0:-1]

    # compute smoothed acceleration by dividing the delta velocity over delta time
    acc = numexpr.evaluate('(vel_matrix_B - vel_matrix_A) * 1000/ (t_matrix_B - t_matrix_A)')
    smoothed_accelaration = np.insert(acc, 0, initial_accel, axis=0)
    
    return np.array(smoothed_velocities), np.array(smoothed_accelaration)

def get_smoothed_x_y_vel_accel(dataset, window):
    smoothed_x_values, smoothed_y_values = get_smoothed_x_y(dataset, window)
    
    
    
    initial_vel = dataset[0, v_vel]
    initial_accel = dataset[0, v_acc]
    time_values = dataset[:, time_column]
    
    smoothed_vel, smoothed_accel = get_smoothed_vel_accel(smoothed_x_values, smoothed_y_values,
                                                          time_values, initial_vel, initial_accel)
    return smoothed_x_values, smoothed_y_values, smoothed_vel, smoothed_accel


def smooth_dataset(window, train, if_vel_acc=None):
    if if_vel_acc:
        smoothed_x_values, smoothed_y_values, smoothed_vel,smoothed_accel = \
            get_smoothed_x_y_vel_accel(train, window)
        train[:, v_vel] = smoothed_vel
        train[:, v_acc] = smoothed_accel
    else:
        smoothed_x_values, smoothed_y_values = get_smoothed_x_y(train, window)
    
    # train[local_x] = [x for x in smoothed_x_values]
    # train[local_y] = [x for x in smoothed_y_values]
    train[:, local_x] = smoothed_x_values
    train[:, local_y] = smoothed_y_values
    
    
    return train
    
def graph_result(init_dataset,kf_dataset,smooth_dataset):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(init_dataset[:, 0], init_dataset[:, 1], color='b', label='ordinary')
    ax.plot(kf_dataset[:, 0], kf_dataset[:, 1], color='y', label='kf')
    ax.plot(smooth_dataset[:, 0], smooth_dataset[:, 1], color='r', label='smooth')#, marker='o', markersize=2)
    ax.grid()
    ax.set_xlabel('x, px')
    ax.set_ylabel('y, px')
    ax.legend()
    plt.show()
    
if __name__ == '__main__':
    
    file_name = '5_轨道 3'
    smoothing_window = 21
    
    curr_path = os.getcwd()
    
    path_to_smoothed = curr_path + "/smoothed"
    if not os.path.isdir(path_to_smoothed):
        os.makedirs(path_to_smoothed)
    
    path_to_dataset = curr_path + f"/{file_name}.npy"
    path_to_smoothed_dataset = path_to_smoothed + f"/{file_name}_smoothed.npy"
    
    train = np.load(path_to_dataset)
    train = train.astype(np.float32)
    
    train_kf = get_kf_x_y(train)
    
    train_smooth=smooth_dataset(smoothing_window,train)
    
    # print(train_smooth)
    graph_result(train,train_kf,train_smooth)
    
    np.save(path_to_smoothed_dataset,train_smooth)
    
    # traj_smooth(file_name, 21)