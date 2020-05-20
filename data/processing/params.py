#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:24:52 2019

@author: luoyao
"""

import os.path as osp
import numpy as np
import pickle
def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

d = "/home/viet/Projects/Pycharm/SPRNet/train.configs/"
keypoints = np.load(osp.join(d, 'keypoints_sim.npy'))
w_shp = np.load(osp.join(d, 'w_shp_sim.npy'))
w_exp = np.load(osp.join(d, 'w_exp_sim.npy'))

meta = pickle.load(open(osp.join(d, 'param_whitening.pkl'), 'rb'))
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
u_shp = np.load(osp.join(d, 'u_shp.npy'))
u_exp = np.load(osp.join(d, 'u_exp.npy'))
u = u_shp + u_exp
w = np.concatenate((w_shp, w_exp), axis=1)  #column join
w_base = w[keypoints]
w_norm = np.linalg.norm(w, axis=0) 
w_base_norm = np.linalg.norm(w_base, axis=0)   #求范数,default 2-norm, axis=0,indicates that coluwn

#for test
dim = w_shp.shape[0]//3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
std_size = 120




























