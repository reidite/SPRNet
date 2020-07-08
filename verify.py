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
from utils.toolkit import show_uv_mesh, get_vertices, estimate_pose, show_kpt_result, UVmap2Mesh, showMesh
import scipy.io as sio
import os.path as osp
import cv2
from math import pi



working_folder  = str(os.path.abspath(os.getcwd()))
FLAGS = {   
			"model"         : os.path.join(working_folder, "train_log/_checkpoint_epoch_80.pth.tar"),
            "data_path"     : os.path.join(working_folder, "data"),
            "data_list"     : os.path.join(working_folder, "test.configs", "AFLW2000-3D.list"),
			"result_path"   : os.path.join(working_folder, "result"),
            "uv_kpt_path"   : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "device"        : "cuda",
            "devices_id"    : [0],
            "batch_size"    : 32, 
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
    NME3DError_List         =   []

    # NME2DError          =   getErrorFunction("NME2D")
    NME3DError          =   getErrorFunction("NME3D")
    with torch.no_grad():
        for i, (img, target) in tqdm(enumerate(data_loader)):
            target.requires_grad  = False
            target = target.cuda(non_blocking=True)
            gens  = model(img, isTrain=False)[:, :, 1:257, 1:257]

            for i in range(gens.shape[0]):
                inp = img[i].cpu().numpy().transpose(1, 2, 0)
                prd = gens[i].cpu().numpy().transpose(1, 2, 0)  * 280.0
                grt = target[i].cpu().numpy().transpose(1, 2, 0) * 280.0

                # NME2DError_List.append(NME2DError(grt, prd))
                NME3DError_List.append(NME3DError(grt, prd))

                show_img    = (255.0 * (inp - np.min(inp))/np.ptp(inp)).astype(np.uint8)
                # P, pose, (s, R, t) = estimate_pose(get_vertices(show_img_uv))
                # if pose[0] >= - 30.0 / 180.0 * pi and pose[0] <=  30.0 / 180.0 * pi:
                #     NME3D_30_Error_List.append(NME3DError(grt, prd))
                # elif (pose[0] >= - 60.0 / 180.0 * pi and pose[0] < - 30.0 / 180.0 * pi) or (pose[0] > 30.0 / 180.0 * pi and pose[0] <= 60.0 / 180.0 * pi):
                #     NME3D_60_Error_List.append(NME3DError(grt, prd))
                # else:
                #     NME3D_90_Error_List.append(NME3DError(grt, prd))
                tex         = cv2.remap(show_img, prd[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
                mesh_info   = UVmap2Mesh(uv_position_map=prd, uv_texture_map=tex)
                showMesh(mesh_info, tex, show_img)
                # show_uv_mesh(show_img, prd)
                # show_kpt_result(show_img, prd, grt)
    # print(np.mean(NME2DError_List))
    print(np.mean(NME3DError_List))
    return

if __name__ == '__main__':
    verify()
