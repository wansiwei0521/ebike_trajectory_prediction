# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-25 13:22:19
LastEditTime: 2022-03-25 14:00:24
LastEditors: Vansw
Description: 
FilePath: //ebike_trajectory_predictiond://MyProject//LocalGit//thesis//trajs_process//kf.py
"""

import numpy as np
from scipy.spatial.distance import euclidean

# def center_base(RS1_x, RS1_y, RS2_x, RS2_y):
#     if(RS1_x < RS2_x):
#         center_base_x = RS1_x + (RS2_x - RS1_x) / 2
#     else: 
#         center_base_x = RS2_x + (RS1_x - RS2_x) / 2
#     if(RS1_y < RS2_y):
#         center_base_y = RS1_y + (RS2_y - RS1_y) / 2
#     else: 
#         center_base_y = RS2_y + (RS1_y - RS2_y) / 2
#     return [center_base_x, center_base_y]

class KalmanFilterConstCoefficient:
    # T0 = 1 # 1 fps
    
    def __init__(self, alpha,fps=1):
        self.T0 = fps
        self.alpha = alpha;
        self.beta = alpha**2 / (2 - alpha)
        
        
    def extrapolate(self, speed, curr):
        a = np.array([[1, self.T0], [0, 1]])
        b = np.array([curr, speed])
        return a @ b

    
    def estimate(self, curr, extra):
        return extra + self.alpha * (curr - extra)


    def speed(self, curr, last, last_speed=0):
        # print(curr,last)
        # print(curr - last)
        return last_speed + self.beta / self.T0 * (curr - last)
    
    
    def filtred_coordinates(self, X, Y):
        last_coor = np.array([X[0], Y[0]])
        curr_coor = np.array([X[1], Y[1]])
        speed_coor = self.speed(curr_coor, last_coor)
        coor = np.array([last_coor, curr_coor])
        for i in range(2, len(X)):
            extra  = self.extrapolate(speed_coor, coor[-1])
            last_coor = curr_coor
            curr_coor = np.array([X[i], Y[i]])
            speed_coor = self.speed(curr_coor, last_coor)
            coor = np.append(coor, [self.estimate(curr_coor, extra[0])], axis=0)
        return coor

# class KalmanFilterVariableCoefficient(KalmanFilterConstCoefficient):
#     def __init__(self, alpha_zone, R_zone, RS1_x, RS1_y, RS2_x, RS2_y):
#         super().__init__(0)
#         self.R_zone = R_zone
#         self.alpha_zone = alpha_zone
#         self.beta_zone = alpha_zone**2 / (2 - alpha_zone)
#         self.center_base_x, self.center_base_y = center_base(RS1_x, RS1_y, RS2_x, RS2_y)
        
        
#     def filtred_coordinates(self, X, Y):
#         last_coor = np.array([X[0], Y[0]])
#         curr_coor = np.array([X[1], Y[1]])
#         R = euclidean([X[1], Y[1]], [self.center_base_x, self.center_base_y])
#         self.alpha = self.alpha_zone[(abs(self.R_zone - R)).argmin()]
#         self.beta = self.beta_zone[(abs(self.R_zone - R)).argmin()]
#         speed_coor = self.speed(curr_coor, last_coor)
#         coor = np.array([last_coor, curr_coor])
#         for i in range(2, len(X)):
#             extra  = self.extrapolate(speed_coor, coor[-1])
#             last_coor = curr_coor
#             curr_coor = np.array([X[i], Y[i]])
#             R = euclidean([X[i], Y[i]], [self.center_base_x, self.center_base_y])
#             self.alpha = self.alpha_zone[(abs(self.R_zone - R)).argmin()]
#             self.beta = self.beta_zone[(abs(self.R_zone - R)).argmin()]
#             speed_coor = self.speed(curr_coor, last_coor)
#             coor = np.append(coor, [self.estimate(curr_coor, extra[0])], axis=0)
#         return coor