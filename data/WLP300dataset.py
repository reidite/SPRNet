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

def create_label_dict_train(path):
	label_dict = defaultdict(list)
	names_list = Path(path).read_text().strip().split('\n')[:100]
	for f_name in names_list:
		f_s = f_name.split('\000')
		label_dict[int(f_s[1])].append(f_s[0])

	return label_dict

def split_label_train(path):
	names_list = Path(path).read_text().strip().split('\n')[:100]
	img_name_nlabel = []
	for img_name in names_list:
		img_name_nlabel.append(img_name.split('\000')[0])
		
	return img_name_nlabel

def create_label_dict_val(path):
	label_dict = defaultdict(list)
	names_list = Path(path).read_text().strip().split('\n')[:100]
	for f_name in names_list:
		f_s = f_name.split('\000')
		label_dict[int(f_s[1])].append(f_s[0])
	
	return label_dict

def split_label_val(path):
	names_list = Path(path).read_text().strip().split('\n')[:100]
	img_name_nlabel = []
	for img_name in names_list:
		img_name_nlabel.append(img_name.split('\000')[0])

	return img_name_nlabel

class SiaTrainDataset(data.Dataset):
	def __init__(self, root_dir, filelists, transform=None):
		self.root_dir 	= root_dir
		self.transform 	= transform
		self.label_dict = create_label_dict_train(filelists)
		self.lines 		= split_label_train(filelists)

	def __getitem__(self, index):
		label_1 = random.choice(range(len(self.label_dict)))
		img1_name = random.choice( self.label_dict[label_1])
		is_same = np.random.choice([0,1], p=[0.8, 0.2])


		if is_same:
			img2_name = random.choice(self.label_dict[label_1])
		else:
			while True:
				label_2 = random.choice(range(len(self.label_dict)))
				if label_2 != label_1:
					break
			img2_name = random.choice( self.label_dict[label_2])
		
		img1_path = osp.join(self.root_dir, "train_aug_120x120", img1_name)
		img2_path = osp.join(self.root_dir, "train_aug_120x120", img2_name)

		target1_path = osp.join(self.root_dir, "train_uv_256x256", img1_name.replace('jpg', 'npy'))
		target2_path = osp.join(self.root_dir, "train_uv_256x256", img2_name.replace('jpg', 'npy'))

		img1 = cv2.imread(img1_path)
		img2 = cv2.imread(img2_path)

		img1 = cv2.resize(img1, None, fx=32/15,fy=32/15,interpolation = cv2.INTER_CUBIC)
		img2 = cv2.resize(img2, None, fx=32/15,fy=32/15,interpolation = cv2.INTER_CUBIC)
		
		uv1  = np.load(target1_path)
		uv2  = np.load(target2_path)

		sample = {'img1': img1, 'img2': img2, 'uv1': uv1, 'uv2': uv2}
		if self.transform:
			sample = self.transform(sample)
		
		img1, img2, uv1, uv2 = sample['img1'], sample['img2'], sample['uv1'], sample['uv2']
		
		return img1, img2, torch.from_numpy(np.array([is_same], dtype = np.float32)), uv1, uv2
	
	def __len__(self):
		return len(self.lines)

class SiaValDataset(data.Dataset):
	def __init__(self, root_dir, filelists, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.label_dict = create_label_dict_val(filelists)
		self.lines = split_label_val(filelists)
		

	def __getitem__(self, index):
		label_1 = random.choice(range(len(self.label_dict)))
		img1_name = random.choice( self.label_dict[label_1])
		is_same = np.random.choice([0,1], p=[0.8, 0.2])


		if is_same:
			img2_name = random.choice(self.label_dict[label_1])
		else:
			while True:
				label_2 = random.choice(range(len(self.label_dict)))
				if label_2 != label_1:
					break
			img2_name = random.choice( self.label_dict[label_2])
		
		img1_path = osp.join(self.root_dir, "train_aug_120x120", img1_name)
		img2_path = osp.join(self.root_dir, "train_aug_120x120", img2_name)

		target1_path = osp.join(self.root_dir, "train_uv_256x256", img1_name.replace('jpg', 'npy'))
		target2_path = osp.join(self.root_dir, "train_uv_256x256", img2_name.replace('jpg', 'npy'))

		img1 = cv2.imread(img1_path)
		img2 = cv2.imread(img2_path)

		img1 = cv2.resize(img1, None, fx=32/15,fy=32/15,interpolation = cv2.INTER_CUBIC)
		img2 = cv2.resize(img2, None, fx=32/15,fy=32/15,interpolation = cv2.INTER_CUBIC)

		uv1  = np.load(target1_path)
		uv2	 = np.load(target2_path)

		sample = {'img1': img1, 'img2': img2, 'uv1': uv1, 'uv2': uv2}
		if self.transform:
			sample = self.transform(sample)
		
		img1, img2, uv1, uv2 = sample['img1'], sample['img2'], sample['uv1'], sample['uv2']
		
		return img1, img2, torch.from_numpy(np.array([is_same], dtype = np.float32)), uv1, uv2
	
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

		uv1 = uv1.astype("float32") / 255.
		uv1 = np.clip(uv1, 0, 1)
		img1 = img1.astype("float32") / 255.
		uv2 = uv2.astype("float32") / 255.
		uv2 = np.clip(uv2, 0, 1)
		img2 = img2.astype("float32") / 255.

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
