import os.path as osp
import numpy as np
import time
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pylab as plt
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from models.siamese_utils import SiaTrainDataset, ToTensor, Normalize
from models.io_utils import mkdir
from models.sia_loss import WPDCLoss_0
from models.resfcn256 import ResFCN256

class sia_net(nn.Module):
	def __init__(self, model):
		super(sia_net, self).__init__()
		self.fc1	=	nn.Sequential(nn.Sequential(*list(model.children())[:-2]), nn.AdaptiveAvgPool2d(1))

		self.fc1_0	=	nn.Sequential(nn.Linear(2048, 1024), nn.Linear(1024, 512))
		self.fc1_1	=	nn.Sequential(nn.Linear(2048, 62))

	def forward_once(self, x):
		x = self.fc1(x)

		x = x.view(x.size()[0], -1)

		feature = self.fc1_0(x)

		param = self.fc1_1(x)

		return feature, param

	def forward(self, input_l, input_r):
		feature_l, param_l = self.forward_once(input_l)
		feature_r, param_r = self.forward_once(input_r)

		return feature_l, feature_r, param_l, param_r

def show_plot(iteration, loss):
	plt.clf()
	plt.ion()
	plt.figure(1)
	plt.plot(iteration, loss, '-r')
	plt.draw()
	time.sleep(0.01)

def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
	assert ep >= 1, "Current epoch number should be >= 1"
	
	if ep < start_decay_at_ep:
		return

	global lr
	lr = base_lr
	for param_group in optimizer.param_groups:
		lr = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
		param_group['lr'] = lr

def load_PRNET():
	prnet = ResFCN256()
	model = sia_net(prnet)