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
from torchsummary import summary
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import AFLWTestDataset, ToTensor, ToNormalize
from models.sia_loss import *
from models.resfcn import load_SPRNET
from utils.toolkit import show_uv_mesh
import scipy.io as sio
import os.path as osp
import os
import cv2
import matplotlib.pylab as plt


working_folder  = str(os.path.abspath(os.getcwd()))
FLAGS = {   
			"model"         : os.path.join(working_folder, "train_log/_checkpoint_epoch_80.pth.tar"),
            "data_path"     : os.path.join(working_folder, "data"),
            "data_list"     : os.path.join(working_folder, "test.configs", "AFLW2000-3D.list"),
			"result_path"   : os.path.join(working_folder, "result"),
            "uv_kpt_path"   : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "device"        : "cuda",
            "devices_id"    : [0],
            "batch_size"    : 16, 
            "workers"       : 8
		}

uv_kpt_ind = np.loadtxt(FLAGS["uv_kpt_path"]).astype(np.int32)


def verify():            
    model   = load_SPRNET()
    torch.cuda.set_device(FLAGS["devices_id"][0])
    model	= nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()
    pretrained_weights = torch.load(FLAGS["model"], map_location='cuda:0')
    model.load_state_dict(pretrained_weights['state_dict'])
    dataset 	        = AFLWTestDataset(
		root_dir  		= FLAGS["data_path"],
		filelists 		= FLAGS["data_list"],
        augmentation    = False,
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
    # summary(model.module.fw, (3, 256, 256))
    NME2DError_List     =   []
    NME3DError_List     =   []

    NME2DError          =   getErrorFunction("NME2D")
    NME3DError          =   getErrorFunction("NME3D")
    with torch.no_grad():
        for i, (img, target) in tqdm(enumerate(data_loader)):
            target.requires_grad  = False
            target = target.cuda(non_blocking=True)
            gens  = model(img, isTrain=False)[:, :, 1:257, 1:257]
            
            for i in range(FLAGS["batch_size"]):
                prd = gens[i].cpu().numpy().transpose(1, 2, 0)  * 280.0
                grt = target[i].cpu().numpy().transpose(1, 2, 0) * 280.0

                NME2DError_List.append(NME2DError(grt, prd))
                NME3DError_List.append(NME3DError(grt, prd))
                
    print(np.mean(NME2DError_List))
    print(np.mean(NME2DError_List))
                # show_img    = img[i].cpu().numpy().transpose(1, 2, 0)
                # show_gen    = gens[i].cpu().numpy().transpose(1, 2, 0)

                # show_img_img    = (255.0 * (show_img - np.min(show_img))/np.ptp(show_img)).astype(int)
                # show_img_uv     = show_gen * 280.0
                # kpt             = show_img_uv[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
                # show_uv_mesh(show_img, show_img_uv, kpt)
    return

if __name__ == '__main__':
    verify()
