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

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
from data.WLP300dataset import SiaTrainDataset, ToTensor, ToNormalize

from models.sia_loss import UVLoss0, WeightMaskLoss
from models.resfcn120 import ResFCN120


FLAGS = {   
			"start_epoch": 1,
			"target_epoch": 500,
			"device": "cuda",
			"mask_path": "./utils/uv_data/uv_weight_mask_gdh.png",
			"batch_size": 32,
			"save_interval": 5,
			"base_lr_exp": 0.001,
			"base_lr_wpdc" : 0.001,
			"base_lr_id" : 0.00005,
			"base_lr_shp" : 0.0001,
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
			"summary_path": "./runs",
			"summary_step": 0,
			"resume": True
		}

lr_exp 	= FLAGS["base_lr_exp"]
lr_id 	= FLAGS["base_lr_id"]
lr_wpdc = FLAGS["base_lr_wpdc"]
lr_shp 	= FLAGS["base_lr_shp"]

LossUV_list, LossWPDC_list, LossSHP_list, LossID_list = [], [], [], []
writer = SummaryWriter(FLAGS['summary_path'])
snapshot = "./train_log/"
# log_mode = 'w'
# resume = ''
# size_average = True
# num_classes = 62
# frozen = 'false'
# task = 'all'
# test_initial = False
# resample_num = 132

class sia_net(nn.Module):
	def __init__(self, model):
		super(sia_net, self).__init__()

		self.sigmoid	= nn.Sequential(nn.Sequential(*list(model.children())[:-1]), nn.Sigmoid())

	def forward_once(self, x):
		uv 		= self.sigmoid(x)
		return uv

	def forward(self, input_l, input_r):
		uv_l = self.forward_once(input_l)
		uv_r = self.forward_once(input_r)

		return uv_l, uv_r
#region TOOLKIT
def save_checkpoint(state, filename="checkpoint.pth.tar"):
	torch.save(state, filename)
	logging.info(f"Save checkpoint to {filename}")

def extract_param(model_params_path):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(model_params_path,map_location=map_location)['state_dict']
    return  checkpoint

def show_plot(iteration, loss):
	plt.clf()
	plt.ion()
	plt.figure(1)
	plt.plot(iteration, loss, '-r')
	plt.draw()
	time.sleep(0.01)

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold', bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

#endregion

#region ADJUST LEARNING RATE
def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
	assert ep >= 1, "Current epoch number should be >= 1"
	
	if ep < start_decay_at_ep:
		return

	global lr_exp
	lr_exp = base_lr
	for param_group in optimizer.param_groups:
		lr_exp = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
		param_group['lr'] = lr_exp

def adjust_lr_id(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    global lr_id
    lr_id = base_lr
    for param_group in optimizer.param_groups:
        lr_id = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
        param_group['lr'] = lr_id

def adjust_lr_wpdc(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
	assert ep >= 1, "Current epoch number should be >= 1"

	if ep < start_decay_at_ep:
		return
	
	global lr_wpdc
	lr_wpdc = base_lr

	for param_group in optimizer.param_groups:
		lr_wpdc = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
		param_group['lr'] = lr_wpdc

def adjust_lr_shp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    global lr_shp
    lr_shp = base_lr
    for param_group in optimizer.param_groups:
        lr_shp = (base_lr*(0.001**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
        param_group['lr'] = lr_shp
#endregion

def load_SPRNET():
	prnet = ResFCN120()
	model = sia_net(prnet)
	
	return model

manualSeed = 5

torch.manual_seed(manualSeed)

#region TRAINING
def train_uv(train_loader, model, criterion_uv, optimizer, epoch):
	model.train()

	for i, (img_l, img_r, label, target_l, target_r) in enumerate(train_loader):
		target_l.requires_grad = False
		target_r.requires_grad = False

		label.requires_grad = False
		label = label.cuda(non_blocking = True)

		target_l = target_l.cuda(non_blocking=True)
		target_r = target_r.cuda(non_blocking=True)

		feature_l, feature_r, param_l, param_r, uv_l, uv_r = model(img_l, img_r)

		loss = criterion_uv(uv_l, uv_r, label, target_l, target_r)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % FLAGS["target_epoch"] == 0:
			print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, FLAGS["lr"], loss.data.cpu().numpy()))
			print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, FLAGS["lr"], loss.data.cpu().numpy()), file=open(FLAGS["log_file"] + 'contrastive_print.txt','a'))
		
		LossUV_list.append(loss.item())

		writer.add_scalar("UV Loss", LossUV_list[-1], FLAGS["summary_step"])


#endregion

def validate(val_loader, model, criterion, epoch):
	model.eval()

	with torch.no_grad():
		losses = []
		for i, (input_l, input_r, label, target_l, target_r) in enumerate(val_loader):

			target_l.requires_grad	=	False
			target_r.requires_grad	=	False

			target_l = target_l.cuda(non_blocking=True)
			target_r = target_r.cuda(non_blocking=True)

			label.requires_grad		=	False
			label = label.cuda(non_blocking=True)

			feature_l, output_l		= model(input_l)
			feature_r, output_r		= model(input_r)

			loss = criterion(output_l, output_r, feature_l, feature_r, label, target_l, target_r)
			loss_cpu				= loss.cpu()
			losses.append(loss_cpu.numpy())
		
		loss	= np.mean(losses)
		print('Testing======>>>[Epoch:%d], loss:%.4f' % (epoch, loss))
		print('[Epoch:%d], loss:%.4f' % (epoch, loss), file=open(log_file + 'test_loss.txt','a'))
		logging.info(
					f'Val: [{epoch}][{len(val_loader)}]\t'
					f'Loss {loss:.4f}\t'
					)

def main(root_dir):
	###	Step1: Define the model structure
	model 	= load_SPRNET()
	torch.cuda.set_device(FLAGS["devices_id"][0])
	model	= nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()

	###	Step2: Loss and optimization method
	criterion = WeightMaskLoss(mask_path=FLAGS["mask_path"])
	optimizer = torch.optim.SGD(model.parameters(),
								lr = FLAGS["base_lr"],
								momentum = FLAGS["momentum"],
								weight_decay = FLAGS["weight_decay"],
								nesterov = True)

	###	Step3: Load 300WLP Dataset
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
		##	Adjust learning rate
		adjust_lr_exp(optimizer, FLAGS["base_lr"], epoch, FLAGS["target_epoch"], 30)
		##	Train for one epoch
		
		train_uv(train_loader, model, criterion, optimizer, epoch)

		
		##	Save model parameters
		filename = f'{snapshot}_checkpoint_epoch_{epoch}.pth.tar'
		save_checkpoint({
							'epoch':epoch,
							'state_dict':model.state_dict()
						},
						filename
						)
	writer.close()

if __name__ == "__main__":
	main(str(os.path.abspath(os.getcwd())))