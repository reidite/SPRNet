import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
from models.sia_loss import *
import numpy as np

class CnvBlock(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding = 0, padding_mode="zeros"):
		super(CnvBlock, self).__init__()
		self.conv 		=	nn.Conv2d(
			in_channels 	= 	in_planes,
			out_channels	= 	out_planes, 
			kernel_size		=	kernel_size, 
			stride			=	stride, 
			padding			=	padding,
			padding_mode	=	padding_mode, 
			bias			=	False
			)
		self.norm		= 	nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.5)
		self.ac 		=	nn.ReLU(inplace=True)
	
	def forward(self, x):
		out				=	self.conv(x)
		out				=	self.norm(out)
		out				=	self.ac(out)
		return out

class TrsBlock(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
		super(TrsBlock, self).__init__()
		self.convT 		=	nn.ConvTranspose2d(
			in_channels 	= 	in_planes,
			out_channels	= 	out_planes, 
			kernel_size		=	kernel_size, 
			stride			=	stride, 
			padding			=	(kernel_size - 1) // 2, 
			output_padding	=	stride - 1, 
			bias			=	False
			)
		self.norm		= 	nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.5)
		self.ac 		=	nn.ReLU(inplace=True)
		self.crop_size 	= 	(kernel_size + 1) % 2

	def forward(self, x):
		up				=	self.convT(x)
		out				=	up[:, :, self.crop_size:up.shape[2], self.crop_size:up.shape[3]].clone()
		out				=	self.norm(out)
		out				=	self.ac(out)
		return out

class ResBlock(nn.Module):
	expansion = 1
	def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, norm_layer=None):
		super(ResBlock, self).__init__()

		self.shortcut_conv  = nn.Conv2d(
			in_planes, 
			out_planes, 
			kernel_size=1,
			stride=stride, 
			bias=False
		) if stride != 1 or in_planes != out_planes else None

		self.conv1 			= CnvBlock(
			in_planes, 
			out_planes // 2, 
			kernel_size = 1, 
			stride = 1, 
			padding = 0
		)
		self.conv2 			= CnvBlock(
			out_planes // 2, 
			out_planes // 2, 
			kernel_size = kernel_size, 
			stride = stride, 
			padding=kernel_size - 1, 
			padding_mode='circular'
		)
		self.conv3 			= nn.Conv2d(
			out_planes // 2, 
			out_planes, 
			stride=1, 
			kernel_size=1, 
			bias=False
		)

		self.stride = stride
		self.out_planes = out_planes

		self.norm		= 	nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.5)
		self.ac 		=	nn.ReLU(inplace=True)
	def forward(self, x):

		shortcut = x

		if self.shortcut_conv:
			shortcut = self.shortcut_conv(x)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)

		x += shortcut
		x = self.norm(x)
		x = self.ac(x)

		return x

class ResFCN256(nn.Module):
	def __init__(self, resolution_input=120, resolution_output=120, channel=3, size=16):
		super().__init__()
		self.input_resolution 	= resolution_input
		self.output_resolution	= resolution_output
		self.channel			= channel
		self.size				= size


		### 256 x 256 x 3
		self.block0				= CnvBlock(in_planes=3, out_planes=self.size, kernel_size=4, stride=1, padding=1)
		
		### 256 x 256 x 16
		self.block1				= ResBlock(in_planes=self.size	  , out_planes=self.size * 2, kernel_size=4, stride=2)
		### 128 x 128 x 32
		self.block2				= ResBlock(in_planes=self.size * 2, out_planes=self.size * 2, kernel_size=4, stride=1)
		### 128 x 128 x 32
		self.block3				= ResBlock(in_planes=self.size * 2, out_planes=self.size * 4, kernel_size=4, stride=2)
		### 64 x 64 x 64
		self.block4				= ResBlock(in_planes=self.size * 4, out_planes=self.size * 4, kernel_size=4, stride=1)
		### 64 x 64 x 64
		self.block5				= ResBlock(in_planes=self.size * 4, out_planes=self.size * 8, kernel_size=4, stride=2)
		### 32 x 32 x 128
		self.block6				= ResBlock(in_planes=self.size * 8, out_planes=self.size * 8, kernel_size=4, stride=1)
		### 32 x 32 x 128
		self.block7				= ResBlock(in_planes=self.size * 8, out_planes=self.size * 16, kernel_size=4, stride=2)
		### 16 x 16 x 256
		self.block8				= ResBlock(in_planes=self.size * 16, out_planes=self.size * 16,kernel_size=4, stride=1)
		### 16 x 16 x 256
		self.block9				= ResBlock(in_planes=self.size * 16, out_planes=self.size * 32, kernel_size=4, stride=2)
		### 8 x 8 x 512
		self.block10			= ResBlock(in_planes=self.size * 32, out_planes=self.size * 32, kernel_size=4, stride=1)
		### 8 x 8 x 512


		self.upsample0			= TrsBlock(self.size * 32, self.size * 32, kernel_size=4, stride=1)
		### 8 x 8 x 512
		self.upsample1			= TrsBlock(self.size * 32, self.size * 16, kernel_size=4, stride=2)
		### 16 x 16 x 256
		self.upsample2			= TrsBlock(self.size * 16, self.size * 16, kernel_size=4, stride=1)
		### 16 x 16 x 256
		self.upsample3			= TrsBlock(self.size * 16, self.size * 16, kernel_size=4, stride=1)
		### 16 x 16 x 256
		self.upsample4			= TrsBlock(self.size * 16, self.size * 8 , kernel_size=4, stride=2)
		### 32 x 32 x 128
		self.upsample5			= TrsBlock(self.size * 8 , self.size * 8 , kernel_size=4, stride=1)
		### 32 x 32 x 128
		self.upsample6			= TrsBlock(self.size * 8 , self.size * 8 , kernel_size=4, stride=1)
		### 32 x 32 x 128
		self.upsample7			= TrsBlock(self.size * 8 , self.size * 4 , kernel_size=4, stride=2)
		### 64 x 64 x 64
		self.upsample8			= TrsBlock(self.size * 4 , self.size * 4 , kernel_size=4, stride=1)
		### 64 x 64 x 64
		self.upsample9			= TrsBlock(self.size * 4 , self.size * 4 , kernel_size=4, stride=1)
		### 64 x 64 x 64
		self.upsample10			= TrsBlock(self.size * 4 , self.size * 2 , kernel_size=4, stride=2)
		### 128 x 128 x 32
		self.upsample11			= TrsBlock(self.size * 2 , self.size * 2 , kernel_size=4, stride=1)
		### 128 x 128 x 32
		self.upsample12			= TrsBlock(self.size * 2 , self.size	 , kernel_size=4, stride=2)
		### 256 x 256 x 16
		self.upsample13			= TrsBlock(self.size	 , self.size     , kernel_size=4, stride=1)
		### 256 x 256 x 16
		self.upsample14			= TrsBlock(self.size	 , self.channel  , kernel_size=4, stride=1)
		### 256 x 256 x 3
		self.upsample15			= TrsBlock(self.channel  , self.channel  , kernel_size=4, stride=1)
		### 256 x 256 x 3
		
		self.upsample16			= nn.ConvTranspose2d(
			in_channels		=	self.channel, 
			out_channels	=	self.channel, 
			kernel_size		=	4,
			stride			=	1,
			padding			=	1,
			output_padding	=	0,
			bias			=	False)
		### 256 x 256 x 3
		self.normalize			= nn.BatchNorm2d(self.channel, eps=0.001, momentum=0.5)
		self.activation			= nn.Tanh()

	def forward(self, x):
		se = self.block0(x)
		
		se = self.block1(se)
		se = self.block2(se)
		se = self.block3(se)
		se = self.block4(se)
		se = self.block5(se)
		se = self.block6(se)
		se = self.block7(se)
		se = self.block8(se)
		se = self.block9(se)
		se = self.block10(se)
		

		pd = self.upsample0(se)
		pd = self.upsample1(pd)
		pd = self.upsample2(pd)
		pd = self.upsample3(pd)
		pd = self.upsample4(pd)
		pd = self.upsample5(pd)
		pd = self.upsample6(pd)
		pd = self.upsample7(pd)
		pd = self.upsample8(pd)
		pd = self.upsample9(pd)
		pd = self.upsample10(pd)
		pd = self.upsample11(pd)
		pd = self.upsample12(pd)
		pd = self.upsample13(pd)
		pd = self.upsample14(pd)
		pd = self.upsample15(pd)
		
		pos= self.upsample16(pd)
		pos= pos[:, :, 1:pos.shape[2], 1:pos.shape[3]]
		pos= self.normalize(pos)
		pos= self.activation(pos)
		return pos

class SPRNet(nn.Module):
	def __init__(self, model):
		super(SPRNet, self).__init__()
		self.fw	= nn.Sequential(*list(model.children()))

	def forward(self, input, isTrain = False):
		if isTrain:
			uv_l = self.fw(input[0])
			uv_r = self.fw(input[1])
			return uv_l, uv_r
		else:
			return self.fw(input)

class Loss():
	def __init__(self):
		self.criterion	=	getLossFunction("FWRSE")
		self.metrics	=	getLossFunction("NME")
	
	def forward(self, posmap, gt_posmap):
		loss_posmap		=	self.criterion(gt_posmap, posmap)
		metrics_posmap 		= 	self.metrics(gt_posmap, posmap)
		return loss_posmap, metrics_posmap

def load_SPRNET():
	prnet = ResFCN256()

	model = SPRNet(prnet)

	return model