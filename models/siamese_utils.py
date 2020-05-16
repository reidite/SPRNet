import os
import torch
import pickle
import argparse
import random
import os.path as os_path
import numpy as np
import torch.utils.data as data

from pathlib import Path
from PIL import Image
from io_utils import  _numpy_to_tensor, _load_cpu, _load_gpu
from params import *
from collections import defaultdict

def create_label_dict(path):
	label_dict = defaultdict(list)
	names_list  = Path(path).read_text().strip().split('\n')
	for f_name in names_list:
		f_s = f_name.split('\000')
		label_dict[int(f_s[1])].append(f_s[0])

	return label_dict

def split_label(path):
	names_list = Path(path).read_text().strip().split('\n')
	img_name_nlabel = []
	for img_name in names_list:
		img_name_nlabel.append(img_name.split('\000')[0])
	
	return img_name_nlabel

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected')

def _parse_param(param):
	p_			= param[:12].reshape(3, -1)
	p 			= p_[:, :3]
	offset		= p_[:, -1].reshape(3, 1)
	alpha_shp	= param[12:52].reshape(-1, 1)
	alpha_exp	= param[52:].reshape(-1, 1)
	return p, offset, alpha_shp, alpha_exp

def read_pairs(pairs_filename):
	pairs		= []
	with open(pairs_filename, 'r') as f:
		for line in f.readline()[1:]:
			pair = line.strip().split()
			pairs.append(pair)

	return np.array(pairs)

def read_pairs_ddfa(pairs_filename):
	pairs		= []
	with open(pairs_filename, 'r') as f:
		for line in f.readlines():
			pair = line.strip().split()
			pairs.append(pair)
	
	return np.array(pairs)

def add_extension(path):
	if os.path.exists(path+'.jpg'):
		return path+'.jpg'
	elif os.path.exists(path+'.jpg'):
		return path+'.jpg'
	else:
		raise RuntimeError('No file "%s" with extension png or jpg.' % path)

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val 	= 0
		self.avg 	= 0
		self.sum 	= 0
		self.count 	= 0

	def update(self, val, n=1):
		self.val	=	val
		self.sum	+= 	val * n
		self.count	+= 	n
		self.avg	=	self.sum / self.count

class ToTensor(object):
	def __call__(self, pic):
		if isinstance(pic, np.ndarray):
			img	= torch.from_numpy(pic.transpose((2, 0, 1)))
			return img.float()
	
	def __repr__(self):
		return self.__class__.__name__ + '()'

class Normalize(object):
	def __init__(self, mean, std):
		self.mean 	= mean
		self.std	= std
	
	def __call__(self, tensor):
		tensor.sub_(self.mean).div_(self.std)
		return tensor

class SiaTrainDataset(data.Dataset):
	def __init__(self, root, filelists, param_fp, transform=None, **kargs):
		self.root		=	root
		self.transform	= 	transform
		self.label_dict	=	create_label_dict(filelists)
		self.lines		=	split_label(filelists)
		self.params		=	_numpy_to_tensor(_load_cpu(param_fp))
		self.img_loader	= 	Image.open

	def _target_loader(self, index):
		target 			= 	self.params[index]
		return target

	def __getitem__(self, index):
		label_1			= random.choice(range(len(self.label_dict)))
		img1_name		= random.choice(self.label_dict[label_1])

		is_same			= np.random.choice([0,1], p=[0.6, 0.4])

		if is_same:
			img2_name	= random.choice(self.label_dict[label_1])
		else:
			while True:
				label_2 = random.choice(range(len(self.label_dict)))
				if label_2 != label_1:
					break
			img2_name = random.choice(self.label_dict[label_2])

		img1_path	= osp.join(self.root, img1_name)
		img2_path	= osp.join(self.root, img2_name)

		img1		= self.img_loader(img1_path)
		img2		= self.img_loader(img2_path)

		index1		= self.lines.index(img1_name)
		index2		= self.lines.index(img2_name)

		target1		= self._target_loader(index1)
		target2		= self._target_loader(index2)

		if self.transform is not None:
			img1	= self.transform(img1)
			img2	= self.transform(img2)

		return img1, img2, torch.from_numpy(np.array([is_same], dtype = np.float32)), target1, target2

	def __len__(self):
		return len(self.lines)

class SiaTestDataset(data.Dataset):
	def __init__(self, filelists, root='', transform=None):
		self.root 		= 	root
		self.transform	=	transform
		self.lines		= 	Path(filelists).read_text().strip().split('\n')
		self.img_loader	= 	Image.open
	
	def __getitem__(self, index):
		path			=	osp.join(self.root, self.lines[index])
		img				=	self.img_loader(path)
		if self.transform is not None:
			img = self.transform(img)
		return img

	def __len__(self):
		return len(self.lines)

class LFW_Pairs_Dataset(data.Dataset):
	def __init__(self, lfw_dir, pairs_txt, transform=None):
		self.transform	=	transform
		self.pairs		=	read_pairs(pairs_txt)
		self.lfw_dir	=	lfw_dir
		self.img_loader	=	Image.open

	def __getitem__(self, index):
		issame 	= None
		pair	= self.pairs[index]
		if len(pair) == 3:
			path0 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
			path1 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
			issame = True
		elif len(pair) == 4:
			path0 = add_extension(os.path.join(self.lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
			path1 = add_extension(os.path.join(self.lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
			issame = False
		if os.path.exists(path0) and os.path.exists(path1):
			img0 = self.img_loader(path0)
			img1 = self.img_loader(path1)
		if self.transform is not None:
			img0 = self.transform(img0)
			img1 = self.transform(img1)
		return img0, img1, issame

	def __len__(self):
		return len(self.pairs)

class DDFA_Pairs_Dataset(data.Dataset):
	def __init__(self, root, pairs_txt, transform=None):
		self.transform = transform
		self.pairs = read_pairs_ddfa(pairs_txt)
		self.root = root
		self.img_loader = Image.open

	def __getitem__(self, index):
		pair = self.pairs[index]
		img0_name = pair[0]
		img1_name = pair[1]
		if int(pair[2]):
			issame = True
		else:
			issame = False
		path0 = os.path.join(self.root, img0_name)
		path1 = os.path.join(self.root, img1_name)

		if os.path.exists(path0) and os.path.exists(path1):
			img0 = self.img_loader(path0)
			img1 = self.img_loader(path1)
		if self.transform is not None:
			img0 - self.transform(img0)
			img1 = self.transform(img1)
		return img0, img1, issame

	def __len__(self):
		return len(self.pairs)

class DDFATestDataset(data.Dataset):
	def __init__(self, filelists, root='', transform=None):
		self.root = root
		self.transform = transform
		self.lines = Path(filelists).read_text().strip().split('\n')
		self.img_loader = Image.open

	def __getitem__(self, index):
		path = osp.join(self.root, self.lines[index])
		img	 = self.img_loader(path)

		if self.transform is not None:
			img = self.transform(img)
		return img

	def __len__(self):
		return len(self.lines)