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
from data.dataset import TestDataset, ToTensor, ToNormalize

from models.sia_loss import *
from models.resfcn import ResFCN256
from models.module import *
import scipy.io as sio
import os.path as osp
import os
import cv2
import matplotlib.pylab as plt


working_folder  = str(os.path.abspath(os.getcwd()))
FLAGS = {   
			"model"         : os.path.join(working_folder, "train_log/_checkpoint_epoch_29.pth.tar"),
            "data_path"     : os.path.join(working_folder, "data/test.data/"),
			"result_path"   : os.path.join(working_folder, "result"),
            "uv_kpt_path"   : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "device"        : "cuda",
            "devices_id"    : [0],
            "batch_size"    : 16, 
            "workers"       : 8
		}

uv_kpt_ind = np.loadtxt(FLAGS["uv_kpt_path"]).astype(np.int32)

class SPRNet(nn.Module):
	def __init__(self, model):
		super(SPRNet, self).__init__()
		self.fw	= nn.Sequential(*list(model.children())[:-1])

	def forward(self, input_l, input_r):
		uv_l = self.fw(input_l)
		uv_r = self.fw(input_r)

		return uv_l, uv_r    


def load_SPRNET():
    prnet = ResFCN256()
    model = SPRNet(prnet)
    return model

def verify():            
    model = load_SPRNET(checkpoint_fp)
    torch.cuda.set_device(FLAGS["devices_id"][0])
    model	= nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()
    pretrained_weights = torch.load(os.path.join(root_dir, "train_log", "_checkpoint_epoch_19.pth.tar"))
    dataset 	        = TestDataset(
		root_dir  		= FLAGS["data_path"],
		filelists 		= os.path.join(root, "train.configs", "label_train_aug_120x120.list.val"),
		transform 		= transforms.Compose([ToTensor()])
	)

    data_loader = DataLoader(
		dataset, 
		batch_size      =   FLAGS["batch_size"], 
		num_workers     =   FLAGS["workers"],
		shuffle         =   False, 
		pin_memory      =   True, 
		drop_last       =   True
	)
    cudnn.benchmark = True   
    model.eval()   

    with torch.no_grad():
        for i, (img_l, img_r, label, target_l, target_r) in enumerate(data_loader):
            target_l.requires_grad  = False
            target_r.requires_grad  = False
            label.requires_grad     = False

            label = label.cuda(non_blocking = True)

            target_l = target_l.cuda(non_blocking=True)
            target_r = target_r.cuda(non_blocking=True)
            uv_l, uv_r = model(img_l, img_r)
            for i in range(16):
                show_img = img_l[i].cpu().numpy().transpose(1, 2, 0)
                show_uv  = uv_l[i].cpu().numpy().transpose(1, 2, 0)

                # kpt_r = uv_r[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
                show_img_img = 255.0 * (show_img - np.min(show_img))/np.ptp(show_img).astype(int)
                show_img_uv  = show_uv * 280.0
                kpt_l = show_img_uv[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
                show_uv_mesh(show_img, show_img_uv, kpt_l)
                # show_uv_mesh(image_path, uv_position_map, kpt)
            return


def show_uv_mesh(img, uv, keypoint):
    img = cv2.resize(img, (256,256))
    x, y, z = uv.transpose(2, 0, 1).reshape(3, -1)
    # img = cv2.resize(img, None, fx=32/15,fy=32/15,interpolation = cv2.INTER_CUBIC)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (255, 0, 0), -1)
    x, y, z = keypoint.transpose().astype(np.int32)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 4, (255, 255, 255), -1)
    # res = cv2.resize(img, None, fx=3,fy=3,interpolation = cv2.INTER_CUBIC)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()

if __name__ == '__main__':
    verify()

