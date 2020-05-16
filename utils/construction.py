import torchvision
import torch
import torch.nn as nn
from collections import defaultdict
import random
from params import *


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

def benchmark_3d_vertex_shp(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex_shp(params[i])
        outputs.append(lm)
    return outputs

def benchmark_3d_vertex(params, dense = True):
	outputs = []
	for i in range(params.shape[0]):
		lm = reconstruct_vertex(params[i], dense)
		outputs.append(lm)
	return outputs

