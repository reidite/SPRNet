import os
import os.path as osp
from pathlib import Path
import numpy as np
import glob
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from collections import defaultdict
import cv2
import pickle
from pathlib import Path
import argparse
import random
from data.augmentation import Augment

def create_label_dict_train(path):
	label_dict = defaultdict(list)
	names_list = Path(path).read_text().strip().split('\n')
	for f_name in names_list:
		f_s = f_name.split('\000')
		label_dict[int(f_s[1])].append(f_s[0])

	return label_dict

def split_label_train(path):
	names_list = Path(path).read_text().strip().split('\n')
	img_name_nlabel = []
	for img_name in names_list:
		img_name_nlabel.append(img_name.split('\000')[0])
		
	return img_name_nlabel

def create_label_dict_val(path):
	label_dict = defaultdict(list)
	names_list = Path(path).read_text().strip().split('\n')
	for f_name in names_list:
		f_s = f_name.split('\000')
		label_dict[int(f_s[1])].append(f_s[0])
	
	return label_dict

def split_label_val(path):
	names_list = Path(path).read_text().strip().split('\n')
	img_name_nlabel = []
	for img_name in names_list:
		img_name_nlabel.append(img_name.split('\000')[0])

	return img_name_nlabel

def DetectFace(data_path, boxScale=1.5) :
	min_size = (120, 120)
	haar_scale = 1.1
	min_neighbors = 3
	haar_flags = 0
	faceCascade = cv2.Load(os.path.join(data_path, "processing/Data/FACE/haarcascade_frontalface_alt.xml"))
	
	img_path_List = glob.glob(os.path.join(data_path, "usr.data", "*.jpg"))
	if len(img_path_List)<=0:
		return None
	for file in img_path_List:
		img = cv2.imread(file)
		gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Equalize the histogram
		cv2.EqualizeHist(gray, gray)
		# Detect the faces
		faces = cv2.HaarDetectObjects(
			gray, faceCascade, cv2.CreateMemStorage(0),
			haar_scale, min_neighbors, haar_flags, min_size
			)
		if faces:
			index = 1
			for face in faces:
				index += 1
	return 

class SiaTrainDataset(data.Dataset):
	def __init__(self, root_dir, filelists, augmentation=False, transform=None):
		self.root_dir 		= root_dir
		self.transform 		= transform
		self.label_dict 	= create_label_dict_train(filelists)
		self.lines 			= split_label_train(filelists)
		self.augmentation	= augmentation

	def __getitem__(self, index):
		label_1 = random.choice(range(len(self.label_dict)))
		img1_name = random.choice( self.label_dict[label_1])
		is_same = np.random.choice([0,1], p=[0.6, 0.4])


		if is_same:
			img2_name = random.choice(self.label_dict[label_1])
		else:
			while True:
				label_2 = random.choice(range(len(self.label_dict)))
				if label_2 != label_1:
					break
			img2_name = random.choice( self.label_dict[label_2])
		
		img1_path = osp.join(self.root_dir, "train_im_256x256", img1_name)
		img2_path = osp.join(self.root_dir, "train_im_256x256", img2_name)

		target1_path = osp.join(self.root_dir, "train_uv_256x256", img1_name.replace('jpg', 'npy'))
		target2_path = osp.join(self.root_dir, "train_uv_256x256", img2_name.replace('jpg', 'npy'))

		img1 = cv2.imread(img1_path)
		img2 = cv2.imread(img2_path)

		uv1  = np.load(target1_path).astype(np.float32)
		uv2  = np.load(target2_path).astype(np.float32)

		### Normalize Img Value 0 ~ 1
		img1 = (img1 / 255.0).astype(np.float32)
		img2 = (img2 / 255.0).astype(np.float32)

		### Random Augmentation
		if self.augmentation:
			img1, uv1	=	Augment(img1, uv1)
			img2, uv2	=	Augment(img2, uv2)
		### Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			img1[:, :, i]	=	(img1[:, :, i] - img1[:, :, i].mean()) / np.sqrt(img1[:, :, i].var() + 0.001)
			img2[:, :, i]	=	(img2[:, :, i] - img2[:, :, i].mean()) / np.sqrt(img2[:, :, i].var() + 0.001)

		### Normalize UV Value 0 ~ 1
		uv1	= uv1 / 280.0
		uv2	= uv2 / 280.0

		sample = {'img1': img1, 'img2': img2, 'uv1': uv1, 'uv2': uv2}
		if self.transform:
			sample = self.transform(sample)

		return sample['img1'], sample['img2'], torch.from_numpy(np.array([is_same], dtype = np.float32)), sample['uv1'], sample['uv2']
	
	def __len__(self):
		return len(self.lines)

class SiaValDataset(data.Dataset):
	def __init__(self, root_dir, filelists, augmentation=False, transform=None):
		self.root_dir 		= root_dir
		self.transform 		= transform
		self.label_dict 	= create_label_dict_val(filelists)
		self.lines 			= split_label_val(filelists)
		self.augmentation	= augmentation

	def __getitem__(self, index):
		label_1 = random.choice(range(len(self.label_dict)))
		img1_name = random.choice( self.label_dict[label_1])
		is_same = np.random.choice([0,1], p=[0.6, 0.4])


		if is_same:
			img2_name = random.choice(self.label_dict[label_1])
		else:
			while True:
				label_2 = random.choice(range(len(self.label_dict)))
				if label_2 != label_1:
					break
			img2_name = random.choice( self.label_dict[label_2])
		
		img1_path = osp.join(self.root_dir, "train_im_256x256", img1_name)
		img2_path = osp.join(self.root_dir, "train_im_256x256", img2_name)

		target1_path = osp.join(self.root_dir, "train_uv_256x256", img1_name.replace('jpg', 'npy'))
		target2_path = osp.join(self.root_dir, "train_uv_256x256", img2_name.replace('jpg', 'npy'))

		img1 = cv2.imread(img1_path)
		img2 = cv2.imread(img2_path)
		
		uv1  = np.load(target1_path).astype(np.float32)
		uv2  = np.load(target2_path).astype(np.float32)

		### Normalize Img Value 0 ~ 1
		img1 = (img1 / 255.0).astype(np.float32)
		img2 = (img2 / 255.0).astype(np.float32)

		### Random Augmentation
		if self.augmentation:
			img1, uv1	=	Augment(img1, uv1)
			img2, uv2	=	Augment(img2, uv2)
		### Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			img1[:, :, i]	=	(img1[:, :, i] - img1[:, :, i].mean()) / np.sqrt(img1[:, :, i].var() + 0.001)
			img2[:, :, i]	=	(img2[:, :, i] - img2[:, :, i].mean()) / np.sqrt(img2[:, :, i].var() + 0.001)

		### Normalize UV Value 0 ~ 1
		uv1	= uv1 / 280.0
		uv2	= uv2 / 280.0

		sample = {'img1': img1, 'img2': img2, 'uv1': uv1, 'uv2': uv2}
		if self.transform:
			sample = self.transform(sample)
		
		return sample['img1'], sample['img2'], torch.from_numpy(np.array([is_same], dtype = np.float32)), sample['uv1'], sample['uv2']
	
	def __len__(self):
		return len(self.lines)

class AFLWTestDataset(data.Dataset):
	def __init__(self, root_dir, filelists, augmentation=False, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.lines = Path(filelists).read_text().strip().split('\n')
		self.augmentation	= augmentation

	def __getitem__(self, index):                          
		img_path 	= osp.join(self.root_dir, "verify_im_256x256", self.lines[index])
		target_path	= osp.join(self.root_dir, "verify_uv_256x256", self.lines[index].replace('jpg', 'npy'))

		img = cv2.imread(img_path)
		uv  = np.load(target_path).astype(np.float32)

		### Normalize Img Value 0 ~ 1
		img = (img / 255.0).astype(np.float32)

		### Random Augmentation
		if self.augmentation:
			img, uv	=	Augment(img, uv)
		### Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			img[:, :, i]	=	(img[:, :, i] - img[:, :, i].mean()) / np.sqrt(img[:, :, i].var() + 0.001)

		### Normalize UV Value 0 ~ 1
		uv	= uv / 280.0

		### Transform to Tensor
		img = img.transpose((2, 0, 1))
		uv = uv.transpose((2, 0, 1))
		return torch.from_numpy(img), torch.from_numpy(uv)

	def __len__(self):
		return len(self.lines)

class USRTestDataset(data.Dataset):
	def __init__(self, root_dir, filelists, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.lines = Path(filelists).read_text().strip().split('\n')

	def __getitem__(self, index):                          
		img_path 	= osp.join(self.root_dir, "verify_im_256x256", self.lines[index])

		img = cv2.imread(img_path)
		uv  = np.load(target_path).astype(np.float32)

		### Normalize Img Value 0 ~ 1
		img = (img / 255.0).astype(np.float32)

		### Normalize Pixel Value For Each RGB Channel
		for i in range(3):
			img[:, :, i]	=	(img[:, :, i] - img[:, :, i].mean()) / np.sqrt(img[:, :, i].var() + 0.001)

		### Transform to Tensor
		img = img.transpose((2, 0, 1))
		return torch.from_numpy(img)

	def __len__(self):
		return len(self.lines)

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		img1, img2, uv1, uv2 = sample['img1'], sample['img2'], sample['uv1'], sample['uv2']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		uv1 = uv1.transpose((2, 0, 1))
		img1 = img1.transpose((2, 0, 1))
		uv2 = uv2.transpose((2, 0, 1))
		img2 = img2.transpose((2, 0, 1))

		return {'img1': torch.from_numpy(img1), 'img2': torch.from_numpy(img2), 'uv1': torch.from_numpy(uv1), 'uv2': torch.from_numpy(uv2)}

class ToNormalize(object):
	"""Normalized process on origin Tensors."""

	def __init__(self, mean, std, inplace=False):
		self.mean = mean
		self.std = std
		self.inplace = inplace
	
	def __call__(self, sample):
		img1, img2, uv1, uv2 = sample['img1'], sample['img2'], sample['uv1'], sample['uv2']
		img1 = F.normalize(img1, self.mean, self.std)
		img2 = F.normalize(img2, self.mean, self.std)
		return {'img1': img1, 'img2': img1, 'uv1': uv1, 'uv2': uv2}

if __name__ == "__main__":
	DetectFace("/home/viet/Projects/Pycharm/SPRNet/data")