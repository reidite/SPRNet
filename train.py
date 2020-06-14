import os
import numpy as np
import time
import logging
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pylab as plt
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import SiaTrainDataset, SiaValDataset, ToTensor

from models.sia_loss import *
from models.resfcn import load_SPRNET, Loss


FLAGS = {   
			"start_epoch": 1,
			"target_epoch": 80,
			"device": "cuda",
			"batch_size": 4,
			"save_interval": 5,
			"base_lr": 2e-4,
			"momentum" : 0.95,
			"weight_decay": 5e-5,
			"milestones" : 15,
			"devices_id" : [0],
			"workers" : 8,
			"log_file" : "./train_log/",
			"images": "./results",
			"gauss_kernel": "original",
			"summary_path": "./runs",
			"summary_step": 0,
			"resume": False
		}

lr 	= FLAGS["base_lr"]

Loss_FWRSE_list, Loss_NME_list = [], []

snapshot = "./train_log/"

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

def plt_imshow(img, one_channel = False):
	if one_channel:
		img = img.mean(dim=0)
	img = img / 2 + 0.5
	npimg = img.numpy()
	if one_channel:
		plt.imshow(npimg, cmap="Greys")
	else:
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
#endregion

#region ADJUST LEARNING RATE	
def adjust_lr(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
	assert ep >= 1, "Current epoch number should be >= 0"
	
	if ep < start_decay_at_ep:
		return

	global lr
	lr = base_lr
	for param_group in optimizer.param_groups:
		lr = (base_lr*(0.0002**(float(ep + 1 - start_decay_at_ep)/(total_ep + 1 - start_decay_at_ep))))
		param_group['lr'] = lr

#endregion

manualSeed = 5

torch.manual_seed(manualSeed)

step = 0
#region TRAIN
def train(train_loader, model, criterion, optimizer, epoch, writer):
	model.train()
	print("[Training...]")
	global Loss_FWRSE_list, Loss_NME_list
	Loss_FWRSE_list = []
	Loss_NME_list 	= []
	for i, (img_l, img_r, label, target_l, target_r) in enumerate(train_loader):
		target_l.requires_grad = False
		target_r.requires_grad = False

		label.requires_grad = False
		label = label.cuda(non_blocking = True)

		target_l = target_l.cuda(non_blocking=True)
		target_r = target_r.cuda(non_blocking=True)

		uv_l, uv_r = model(input=[img_l, img_r], isTrain=True)

		loss_l, metrics_l, loss_r, metrics_r  = criterion.forward(uv_l[:, :, 1:257, 1:257], target_l, label, uv_r[:, :, 1:257, 1:257], target_r)
		optimizer.zero_grad()
		loss_l.backward()
		optimizer.step()

		optimizer.zero_grad()
		loss_r.backward()
		optimizer.step()

		global step
		step = epoch * len(train_loader) + i
		Loss_FWRSE_list.append(loss_l.item())
		Loss_NME_list.append(metrics_l.item())
		print('<L>[Step:%d | Epoch:%d], lr:%.6f, loss FWRSE:%.6f, loss NME:%.6f' % (i, epoch, lr, Loss_FWRSE_list[-1], Loss_NME_list[-1]))
		Loss_FWRSE_list.append(loss_r.item())
		Loss_NME_list.append(metrics_r.item())
		print('<R>[Step:%d | Epoch:%d], lr:%.6f, loss FWRSE:%.6f, loss NME:%.6f' % (i, epoch, lr, Loss_FWRSE_list[-1], Loss_NME_list[-1]))
	writer.add_scalar("Loss FWRSE", np.mean(Loss_FWRSE_list), epoch)
	writer.add_scalar("Loss NME", np.mean(Loss_NME_list), epoch)
		# if i % FLAGS["target_epoch"] == 0:
		# 	print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, FLAGS["lr"], total_loss.data.cpu().numpy()))
		# 	print('[Step:%d | Epoch:%d], lr:%.6f, loss:%.6f' % (i, epoch, FLAGS["lr"], total_loss.data.cpu().numpy()), file=open(FLAGS["log_file"] + 'contrastive_print.txt','a'))

#endregion

#region VALIDATE
def validate(val_loader, model, criterion, optimizer, epoch, writer):
	model.eval()
	global Loss_FWRSE_list, Loss_NME_list
	Loss_FWRSE_list = []
	Loss_NME_list 	= []
	with torch.no_grad():
		losses = []
		print("[Validating...]")
		for i, (img_l, img_r, label, target_l, target_r) in enumerate(val_loader):
			target_l.requires_grad = False
			target_r.requires_grad = False

			label.requires_grad = False
			label = label.cuda(non_blocking = True)

			target_l = target_l.cuda(non_blocking=True)
			target_r = target_r.cuda(non_blocking=True)

			uv_l, uv_r = model(input=[img_l, img_r], isTrain=True)

			loss_l, metrics_l, loss_r, metrics_r = criterion.forward(uv_l[:, :, 1:257, 1:257], target_l, label, uv_r[:, :, 1:257, 1:257], target_r)

			Loss_FWRSE_list.append(loss_l.item())
			Loss_NME_list.append(metrics_l.item())
			print('<L>[Step:%d | Epoch:%d], loss FWRSE:%.6f, loss NME:%.6f' % (i, epoch, Loss_FWRSE_list[-1], Loss_NME_list[-1]))
			Loss_FWRSE_list.append(loss_r.item())
			Loss_NME_list.append(metrics_r.item())
			print('<R>[Step:%d | Epoch:%d], loss FWRSE:%.6f, loss NME:%.6f' % (i, epoch, Loss_FWRSE_list[-1], Loss_NME_list[-1]))
		writer.add_scalar("Loss FWRSE", np.mean(Loss_FWRSE_list), epoch)
		writer.add_scalar("Loss NME", np.mean(Loss_NME_list), epoch)

#endregion

def main(root_dir):
	###	Step1: Define the model structure
	model 	= load_SPRNET()
	torch.cuda.set_device(FLAGS["devices_id"][0])
	model	= nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()
	
	if FLAGS["resume"]:
		pretrained_weights = torch.load(os.path.join(root_dir, "train_log", "_checkpoint_epoch_19.pth.tar"))
		model.load_state_dict(pretrained_weights['state_dict'])
		FLAGS["start_epoch"] = int(pretrained_weights['epoch']) + 1
	###	Step2: Loss and optimization method
	criterion = Loss()
	optimizer = torch.optim.SGD(model.parameters(),
								lr 				= FLAGS["base_lr"],
								momentum 		= FLAGS["momentum"],
								weight_decay 	= FLAGS["weight_decay"],
								nesterov 		= True)

	###	Step3: Load 300WLP Augmentation Dataset
	data_dir 		= os.path.join(root_dir, "data")
	train_dataset 	= SiaTrainDataset(
		root_dir  		= data_dir,
		filelists 		= os.path.join(root_dir, "train.configs", "label_train_aug_120x120.list.train"),
		augmentation	= True,
		transform 		= transforms.Compose([ToTensor()])
	)

	val_dataset 	= SiaValDataset(
		root_dir  		= data_dir,
		filelists 		= os.path.join(root_dir, "train.configs", "label_train_aug_120x120.list.val"),
		augmentation	= True,
		transform 		= transforms.Compose([ToTensor()])
	)

	train_loader 	= DataLoader(
		train_dataset, 
		batch_size 		= FLAGS["batch_size"], 
		num_workers		= FLAGS["workers"],
		shuffle			= True, 
		pin_memory		= True,
		drop_last		= True
	)

	val_loader 		= DataLoader(
		val_dataset, 
		batch_size 		= FLAGS["batch_size"], 
		num_workers		= FLAGS["workers"],
		shuffle			= True, 
		pin_memory		= True, 
		drop_last		= True
	)

	cudnn.benchmark = True
	###	Step4: Setup TensorBoard
	train_writer 			= SummaryWriter("runs/train")
	val_writer				= SummaryWriter("runs/val")
	x 			= torch.rand(1, 3, 256, 256).cuda()
	y 			= torch.rand(1, 3, 256, 256).cuda()

	train_writer.add_graph(model.module.fw, x)

	###	Step5: Show CUDA Info
	os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS["devices_id"][0])
	print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name(0))
	for epoch in range(FLAGS["start_epoch"], FLAGS["target_epoch"]):
		print("[Epoch]: %d" %epoch)
		##	Adjust learning rate
		adjust_lr(optimizer, FLAGS["base_lr"], epoch, FLAGS["target_epoch"], FLAGS["milestones"])
		##	Train
		train(train_loader, model, criterion, optimizer, epoch, train_writer)
		##	Validate
		validate(val_loader, model, criterion, optimizer, epoch, val_writer)

		## Step5: Save model parameters
		if (epoch) % 5 == 0:
			filename = f'{snapshot}_checkpoint_epoch_{epoch}.pth.tar'
			save_checkpoint({
						'epoch':epoch,
						'state_dict':model.state_dict()																																																																															
					},
					filename				
					)
	train_writer.close()
	val_writer.close()

if __name__ == "__main__":
	main(str(os.path.abspath(os.getcwd())))
