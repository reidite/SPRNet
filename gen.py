import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
from models.resfcn import load_SPRNET
from data.dataset import USRTestDataset, ToTensor
from torch.utils.data import DataLoader

working_folder  = "/home/viet/Projects/Pycharm/SPRNet/"
FLAGS = {   
            "model"             : os.path.join(working_folder, "train_log/_checkpoint_epoch_80.pth.tar"),
            "data_path"         : os.path.join(working_folder, "data"),
			"uv_kpt_ind_path"   : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "face_ind_path"     : os.path.join(working_folder, "data/processing/Data/UV/face_ind.txt"),
            "triangles_path"    : os.path.join(working_folder, "data/processing/Data/UV/triangles.txt"),
			"result_path"       : os.path.join(working_folder, "result/usr"),
            "uv_kpt_path"       : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "device"            : "cuda",
            "devices_id"        : [0],
            "batch_size"        : 16, 
            "workers"           : 8
		}

def gen():
    model               = load_SPRNET()
    torch.cuda.set_device(FLAGS["devices_id"][0])
    model	            = nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()
    pretrained_weights  = torch.load(FLAGS["model"], map_location='cuda:0')
    model.load_state_dict(pretrained_weights['state_dict'])
    dataset 	        = USRTestDataset(
		root_dir  		= FLAGS["data_path"],
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
        for i, (img) in tqdm(enumerate(data_loader)):
            gens  = model(img, isTrain=False)[:, :, 1:257, 1:257]