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
from models.sia_loss import UVLoss0, WeightMaskLoss
from models.resfcn256 import ResFCN256



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
		self.fc1	=	nn.Sequential(*list(model.children())[:-1])

		# self.fc1_0	=	nn.Sequential(nn.Linear(2048, 1024), nn.Linear(1024, 512))

		# self.fc1_1	=	nn.Sequential(nn.Linear(2048, 62))

	def forward_once(self, x):
		
		# x = self.fc1(x)

		# x = x.view(x.size()[0], -1)

		# feature = self.fc1_0(x)

		uv = self.fc1(x)
		feature = self.fc1(x)

		return feature, uv

	def forward(self, input_l, input_r):
		feature_l, uv_l = self.forward_once(input_l)
		feature_r, uv_r = self.forward_once(input_r)

		return feature_l, feature_r, uv_l, uv_r

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

def load_SPRNET():
	prnet = ResFCN256()
	model = sia_net(prnet)
	
	return model

def save_checkpoint(state, filename="checkpoint.pth.tar"):
	torch.save(state, filename)
	logging.info(f"Save checkpoint to {filename}")

manualSeed = 5

torch.manual_seed(manualSeed)

FLAGS = {   
			"start_epoch": 1,
			"target_epoch": 500,
			"device": "cuda",
			"mask_path": "./utils/uv_data/uv_weight_mask_gdh.png",
			"lr": 0.0001,
			"batch_size": 32,
			"save_interval": 5,
			"base_lr": 0.001,
			"momentum" : 0.9,
			"weight_decay": 5e-4,
			"epochs": 50,
			"milestones" : 30,
			"print_freq" : 50,
			"devices_id" : [0],
			"workers" : 8,
			"log_file" : "./training_debug/logs/TEST_Git/",
			"normalize_mean": [0.485, 0.456, 0.406],
			"normalize_std": [0.229, 0.224, 0.225],
			"images": "./results",
			"gauss_kernel": "original",
			"summary_path": "./prnet_runs",
			"summary_step": 0,
			"resume": True
		}

def train(train_loader, model, criterion, optimizer, epoch):
	model.train()
	for i, (img_l, img_r, label, target_l, target_r) in enumerate(train_loader):
		target_l.requires_grad = False
		target_r.requires_grad = False

		label.requires_grad = False
		label = label.cuda(non_blocking = True)

		target_l = target_l.cuda(non_blocking=True)
		target_r = target_r.cuda(non_blocking=True)

		feature_l, feature_r, output_l, output_r = model(img_l, img_r)

		loss = criterion(output_l, output_r, label, target_l, target_r)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % FLAGS["target_epoch"] == 0:
			print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, FLAGS["lr"], loss.data.cpu().numpy()))
			print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, FLAGS["lr"], loss.data.cpu().numpy()), file=open(FLAGS["log_file"] + 'contrastive_print.txt','a'))

def main(root_dir):
	###		Step1: Define the model structure
	model 	= load_SPRNET()
	torch.cuda.set_device(FLAGS["devices_id"][0])
	model	= nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()

	###		Step2: Loss and optimization method
	criterion = WeightMaskLoss(mask_path=FLAGS["mask_path"])
	optimizer = torch.optim.SGD(model.parameters(),
								lr = FLAGS["base_lr"],
								momentum = FLAGS["momentum"],
								weight_decay = FLAGS["weight_decay"],
								nesterov = True)

	#		Step3: Data
	train_dataset = SiaTrainDataset(
		root_dir  = root_dir,
		transform = transforms.Compose([ToTensor(), ToNormalize(FLAGS["normalize_mean"], FLAGS["normalize_std"])])
	)

	transform_img = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(dir, FLAGS["normalize_std"])
	])

	train_loader = DataLoader(train_dataset, batch_size = FLAGS["batch_size"], num_workers=FLAGS["workers"],
								shuffle=False, pin_memory=True, drop_last=True)


	cudnn.benchmark = True

	for epoch in range(FLAGS["start_epoch"], FLAGS["target_epoch"]):
		#adjust learning rate
		adjust_lr_exp(optimizer, FLAGS["base_lr"], epoch, FLAGS["target_epoch"], 30)
		#train for one epoch
		train(train_loader, model, criterion, optimizer, epoch)
		#save model paramers
		filename = f'{snapshot}_checkpoint_epoch_{epoch}.pth.tar'
		save_checkpoint({
							'epoch':epoch,
							'state_dict':model.state_dict()
						},
						filename
						)

if __name__ == "__main__":
	main(str(os.path.abspath(os.getcwd())))