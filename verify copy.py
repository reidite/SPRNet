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
from data.dataset import SiaTrainDataset, SiaValDataset, ToTensor, ToNormalize

from models.sia_loss import *
from models.resfcn import ResFCN256
from models.module import *
import scipy.io as sio
import os.path as osp
import os
import cv2
import matplotlib.pylab as plt

uv_kpt_ind = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/uv_kpt_ind.txt").astype(np.int32)

class SPRNet(nn.Module):
	def __init__(self, model):
		super(SPRNet, self).__init__()
		self.fw	= nn.Sequential(*list(model.children())[:-1])

	def forward(self, input_l, input_r):
		uv_l = self.fw(input_l)
		uv_r = self.fw(input_r)

		return uv_l, uv_r    
class InitLoss(nn.Module):
    def __init__(self):
        super(InitLoss, self).__init__()
        self.criterion = getLossFunction('FWRSE')
        self.metrics = getLossFunction('NME')

    def forward(self, posmap, gt_posmap):
        loss_posmap = self.criterion(gt_posmap, posmap)
        total_loss = loss_posmap
        metrics_posmap = self.metrics(gt_posmap, posmap)
        return total_loss, metrics_posmap

class InitPRN2(nn.Module):
    def __init__(self):
        super(InitPRN2, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size
        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)  # 256 x 256 x 16
        self.encoder = nn.Sequential(
            PRNResBlock(in_channels=feature_size, out_channels=feature_size * 2, kernel_size=4, stride=2, with_conv_shortcut=True),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1, with_conv_shortcut=False),  # 128 x 128 x 32
            PRNResBlock(in_channels=feature_size * 2, out_channels=feature_size * 4, kernel_size=4, stride=2, with_conv_shortcut=True),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1, with_conv_shortcut=False),  # 64 x 64 x 64
            PRNResBlock(in_channels=feature_size * 4, out_channels=feature_size * 8, kernel_size=4, stride=2, with_conv_shortcut=True),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1, with_conv_shortcut=False),  # 32 x 32 x 128
            PRNResBlock(in_channels=feature_size * 8, out_channels=feature_size * 16, kernel_size=4, stride=2, with_conv_shortcut=True),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1, with_conv_shortcut=False),  # 16 x 16 x 256
            PRNResBlock(in_channels=feature_size * 16, out_channels=feature_size * 32, kernel_size=4, stride=2, with_conv_shortcut=True),  # 8 x 8 x 512
            PRNResBlock(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1, with_conv_shortcut=False),  # 8 x 8 x 512
        )
        self.decoder = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4, stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4, stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4, stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4, stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),  # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Tanh())
        )
        self.loss = InitLoss()

    def forward(self, img, target):
        x = self.layer0(img)
        x = self.encoder(x)
        x = self.decoder(x)
        loss, metrics = self.loss(x, target)
        return loss, metrics, x

def load_SPRNET(checkpoint):
    visible_gpus = '0,1,2,3'
    gpus = visible_gpus.split(',')
    visible_devices = [int(i) for i in gpus]
    prnet = InitPRN2()
    device_ids = [0]
    torch.cuda.set_device(device_ids[0])      
    prnet = nn.DataParallel(prnet, device_ids=device_ids)
    prnet.to(torch.device("cuda:" + gpus[0] if torch.cuda.is_available() else "cpu"))
    prnet.module.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))
    model = SPRNet(prnet)

    return model


def transform_for_infer(image_shape):
    return transforms.Compose(
        [transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )


def SPRNET_validate_ddfa(checkpoint_fp, root, log_dir, device_ids = [0], batch_size = 16, num_workers = 4):
    # checkpoint = torch.load(checkpoint_fp, map_location=map_location)                  
    model = load_SPRNET(checkpoint_fp)    #get a model explain or document
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # model.load_state_dict(checkpoint)
    data_dir 		= "/media/viet/Vincent/SPRNet"
    dataset 	= SiaValDataset(
		root_dir  		= data_dir,
		filelists 		= os.path.join(root, "train.configs", "label_train_aug_120x120.list.val"),
		transform 		= transforms.Compose([ToTensor()])
	)

    data_loader = DataLoader(
		dataset, 
		batch_size=16, 
		num_workers=4,
		shuffle=False, 
		pin_memory=True, 
		drop_last=True
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
    ################## DDFA #####################
    checkpoint_fp = "/home/viet/Projects/Pycharm/SPRNet/train_log/best.pth"
    root = "/home/viet/Projects/Pycharm/SPRNet/"
    log_dir = "/home/viet/Projects/Git/Siamese-3DFace/recognition/"

    SPRNET_validate_ddfa(checkpoint_fp, root, log_dir)
