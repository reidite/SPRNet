import os
import os.path as osp
from pathlib import Path
import numpy as np
import glob
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import cv2
import pickle
import argparse
import random

def create_label_dict(path):
	"""
	Return list of picture's name in group of identity
	"""
	label_list = []
	folder_list = []
	folder_num = 0
	for folder in glob.glob(os.path.join(path, "data", "origin", "*")):
		folder_list += [folder]
	folder_list.sort(key = lambda x: (int(os.path.basename(os.path.normpath(x))), x))
	for folder in folder_list:
		identity_list = []
		folder_num += 1
		for file in glob.glob(os.path.join(folder, "*.npy")):
			identity_list = identity_list + [osp.basename(osp.splitext(file)[0])]
		label_list.append(identity_list)
	return label_list, folder_num


class SiaTrainDataset(data.Dataset):
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.label_dict, self.numb_label = create_label_dict(root_dir)

	def __getitem__(self, index):
		label_1 = random.choice(range(self.numb_label))
		label_2 = -1
		if label_1 == 1991:
			i = 0
		img1_name = self.label_dict[label_1][random.choice(range(len(self.label_dict[label_1])))]
		
		is_same = np.random.choice([0, 1], p=[0.6, 0.4])

		if is_same:
			label_2 = label_1
			img2_name = self.label_dict[label_2][random.choice(range(len(self.label_dict[label_2])))]
		else:
			while True:
				label_2 = random.choice(range(self.numb_label))
				if label_2 != label_1:
					break
			img2_name = self.label_dict[label_2][random.choice(range(len(self.label_dict[label_2])))]
		
		is_flip = np.random.choice([0, 1], p=[0.6, 0.4])
		if is_flip:
			img1_path 		= osp.join(self.root_dir, "data", "flip", str(label_1), img1_name + ".jpg")
			target1_path 	= osp.join(self.root_dir, "data", "flip", str(label_1), img1_name + ".npy")
		else:
			img1_path 		= osp.join(self.root_dir, "data", "origin", str(label_1), img1_name + ".jpg")
			target1_path 	= osp.join(self.root_dir, "data", "origin", str(label_1), img1_name + ".npy")

		is_flip = np.random.choice([0, 1], p=[0.6, 0.4])
		if is_flip:
			img2_path 		= osp.join(self.root_dir, "data", "flip", str(label_2), img2_name + ".jpg")
			target2_path 	= osp.join(self.root_dir, "data", "flip", str(label_2), img2_name + ".npy")
		else:
			img2_path 		= osp.join(self.root_dir, "data", "origin", str(label_2), img2_name + ".jpg")
			target2_path 	= osp.join(self.root_dir, "data", "origin", str(label_2), img2_name + ".npy")
		
		img1 = cv2.imread(img1_path)
		img2 = cv2.imread(img2_path)
		uv1  = np.load(target1_path)
		uv2	 = np.load(target2_path)

		sample = {'img1': img1, 'img2': img2, 'uv1': uv1, 'uv2': uv2}
		if self.transform:
			sample = self.transform(sample)
		
		img1, img2, uv1, uv2 = sample['img1'], sample['img2'], sample['uv1'], sample['uv2']
		
		return img1, img2, torch.from_numpy(np.array([is_same], dtype = np.float32)), uv1, uv2
	
	def __len__(self):
		train_length = 0
		for i in range(self.numb_label):
			train_length += len(self.label_dict[i])
		return train_length

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
