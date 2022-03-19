# coding=utf-8
"""
Author: Vansw
Email: wansiwei1010@163.com
Date: 2022-03-18 11:05:50
LastEditTime: 2022-03-18 11:13:41
LastEditors: Vansw
Description: storing reward function weights
FilePath: //Preference-Planning-Deep-IRLd://MyProject//LocalGit//thesis//something done//param_store.py
"""
import tensorlayer as tl

def save_weights(reward_func, path):  # save trained weights
    tl.files.save_npz(reward_func.trainable_weights, name=path+'/model_q_net1.npz')
    
def load_weights(reward_func, path):  # load trained weights
    tl.files.load_and_assign_npz(name=path+'/model_q_net1.npz', network=reward_func)
