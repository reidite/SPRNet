import os
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
from data.WLP300dataset import SiaTrainDataset, ToTensor, ToNormalize
# from io_utils import mkdir
# from sia_loss import WPDCLoss_0
from models.resfcn256 import ResFCN256

#global configuration
lr = None
#arch
start_epoch = 1
param_fp_train='./train.configs/param_all_norm.pkl'     
param_fp_val='./train.configs/param_all_norm_val.pkl' 
warmup = 5
#opt_style 
batch_size = 32
base_lr = 0.001
lr = base_lr
momentum = 0.9
weight_decay = 5e-4
epochs = 50
milestones = 30, 40
print_freq = 50
devices_id = [0]
workers = 8
filelists_train = "./label_train_aug_120x120.list.train"
filelists_val = "./label_train_aug_120x120.list.val"
root = "/home/luoyao/Project_3d/3D_face_solution/3DDFA_TPAMI/3DDFA_PAMI/train_aug_120x120" 
log_file = "./training_debug/logs/TEST_Git/"
#loss
snapshot = "./training_debug/logs/TEST_Git/"
log_mode = 'w'
resume = ''
size_average = True
num_classes = 62
frozen = 'false'
task = 'all'
test_initial = False
resample_num = 132

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
	
	return model

def save_checkpoint(state, filename="checkpoint.pth.tar"):
	torch.save(state, filename)
	logging.info(f"Save checkpoint to {filename}")

manualSeed = 5

torch.manual_seed(manualSeed)

FLAGS = {
	"lr"  : None,
	"start_epoch": 1,
	"device": "cuda",

	"batch_size" : 32,
	"base_lr": 0.001,
	"momentum" : 0.9,
	"weight_decay": 5e-4,
	"epochs": 50,
	"milestones" : 30
}
FLAGS = {   
			"start_epoch": 0,
			"target_epoch": 500,
			"device": "cuda",
			"lr": 0.0001,
			"batch_size": 16,
			"save_interval": 5,
			"normalize_mean": [0.485, 0.456, 0.406],
			"normalize_std": [0.229, 0.224, 0.225],
			"images": "./results",
			"gauss_kernel": "original",
			"summary_path": "./prnet_runs",
			"summary_step": 0,
			"resume": True
		}
def main(root_dir):
	train_dataset = SiaTrainDataset(
		root_dir  = root_dir,
		transform = transforms.Compose([ToTensor(), ToNormalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])])
	)

	transform_img = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])
	])
	start_epoch, target_epoch = FLAGS['start_epoch'], FLAGS['target_epoch']
	model = load_PRNET()
	torch.cuda.set_device(devices_id[0])
	model	= nn.DataParallel(model, devices_ids=devices_id).cuda()
	train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=workers,
								shuffle=False, pin_memory=True, drop_last=True)


if __name__ == "__main__":
	main(str(os.path.abspath(os.getcwd())))