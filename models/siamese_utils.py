import os
import torch
import pickle
import argparse
import random
import os.path as os_path
import numpy as np
import torch.utils.data as data

from pathlib import Path
from PIL import Imagesiamese_utils
from models.io_utils import  _numpy_to_tensor, _load_cpu, _load_gpu
from models.params import *
from collections import defaultdict

def reconstruct_vertex(param, whitening=True, dense=True):
	if len(param) == 12:
		param = np.concatenate((param, [0] * 50))
	if whitening:
		if len(param) == 62:
			param = param * param_std + param_mean
		else:
			param = np.concatenate((param[:11], [0], param[11:]))
			param = param * param_std + param_mean

	p_			= param[:12].reshape(3, -1)
	p 			= p_[:, :3]
	offset		= p_[:, -1].reshape(3, 1)
	alpha_shp	= param[12:52].reshape(-1, 1)
	alpha_exp	= param[52:].reshape(-1, 1)

	if dense:
		vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
	else:                                                          # reshape(3, -1, order='F') order='F'竖着读，竖着写，优先读/写一列
		"""For 68 pts"""  # get 68 keypoint 3d position  p:3x3 (u + w_shp...):159645x1--->3x53215
		vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset   
		# for landmarks
		vertex[1, :] = std_size + 1 - vertex[1, :]

	return vertex

def reconstruct_vertex_shp(param, whitening=True, dense=True):
	if len(param) == 12:
		param = np.concatenate((param, [0] * 50))
	
	if whitening:
		if len(param) == 62:
			param = param * param_std + param_mean
		else:
			param = np.concatenate((param[:11], [0], param[11:]))
			param = param * param_std + param_mean

	p_			= param[:12].reshape(3, -1)
	p 			= p_[:, :3]
	offset		= p_[:, -1].reshape(3, 1)
	alpha_shp	= param[12:52].reshape(-1, 1)
	alpha_exp	= param[52:].reshape(-1, 1)

	if dense:
		vertex  = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
	else:
		vertex	= (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F')
		vertex[1, :] = std_size + 1 - vertex[1, :]
	
	return vertex

def create_label_dict(path):
	label_dict = defaultdict(list)
	names_list  = Path(path).read_text().strip().split('\n')
	for f_name in names_list:
		f_s = f_name.split('\000')
		label_dict[int(f_s[1])].append(f_s[0])

	return label_dict

def split_label(path):
	names_list = Path(path).read_text().strip().split('\n')
	img_name_nlabel = []
	for img_name in names_list:
		img_name_nlabel.append(img_name.split('\000')[0])
	
	return img_name_nlabel

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected')
	