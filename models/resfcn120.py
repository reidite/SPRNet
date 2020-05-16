import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

import numpy as np



def conv3x3(in_planes, out_planes, stride=1, dilation=1,padding="same"):
	if padding == "same":
		return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=dilation)

class ResBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, kernel_size=3, norm_layer=None):
		super(ResBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d

		self.shortcut_conv  = nn.Conv2d(inplanes, planes, kernel_size=1,stride=stride)
		self.conv1 			= nn.Conv2d(inplanes, planes // 2, kernel_size = 1, stride = 1, padding = 0)
		self.conv2 			= nn.Conv2d(planes // 2, planes // 2, kernel_size = kernel_size, stride = stride, padding=kernel_size//2)
		self.conv3 			= nn.Conv2d(planes // 2, planes, kernel_size = 1, stride = 1, padding = 0)

		self.normalizer_fn = norm_layer(planes)
		self.activation_fn = nn.ReLU(inplace=True)
		self.stride = stride
		self.out_planes = planes
	
	def forward(self, x):
		shortcut = x
		(_, _, _, x_planes) = x.size()

		if self.stride != 1 or x_planes != self.out_planes:
			shortcut = self.shortcut_conv(x)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)

		x += shortcut
		x = self.normalizer_fn(x)
		x = self.activation_fn(x)

		return x

class ResFCN256(nn.Module):
	def __init__(self, resolution_input=120, resolution_output=120, channel=3, size=16):
		super().__init__()
		self.input_resolution 	= resolution_input
		self.output_resolution	= resolution_output
		self.channel			= channel
		self.size				= size


		### 256 x 256 x 3
		self.block0				= conv3x3(in_planes=3, out_planes=self.size, padding="same")
		### 256 x 256 x 16
		self.block1				= ResBlock(inplanes=self.size	 , planes=self.size * 2, stride=2)
		### 128 x 128 x 32
		self.block2				= ResBlock(inplanes=self.size * 2, planes=self.size * 2, stride=1)
		### 128 x 128 x 32
		self.block3				= ResBlock(inplanes=self.size * 2, planes=self.size * 4, stride=2)
		### 64 x 64 x 64
		self.block4				= ResBlock(inplanes=self.size * 4, planes=self.size * 4, stride=1)
		### 64 x 64 x 64
		self.block5				= ResBlock(inplanes=self.size * 4, planes=self.size * 8, stride=2)
		### 32 x 32 x 128
		self.block6				= ResBlock(inplanes=self.size * 8, planes=self.size * 8, stride=1)
		### 32 x 32 x 128
		self.block7				= ResBlock(inplanes=self.size * 8, planes=self.size * 16, stride=2)
		### 16 x 16 x 256
		self.block8				= ResBlock(inplanes=self.size * 16, planes=self.size * 16,stride=1)
		### 16 x 16 x 256
		self.block9				= ResBlock(inplanes=self.size * 16, planes=self.size * 32, stride=2)
		### 8 x 8 x 512
		self.block10			= ResBlock(inplanes=self.size * 32, planes=self.size * 32, stride=1)
		### 8 x 8 x 512


		self.upsample0			= nn.ConvTranspose2d(self.size * 32, self.size * 32, kernel_size=3, stride=1, padding=1)
		### 8 x 8 x 512
		self.upsample1			= nn.ConvTranspose2d(self.size * 32, self.size * 16, kernel_size=4, stride=2, padding=1)
		### 16 x 16 x 256
		self.upsample2			= nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
		### 16 x 16 x 256
		self.upsample3			= nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
		### 16 x 16 x 256
		self.upsample4			= nn.ConvTranspose2d(self.size * 16, self.size * 8 , kernel_size=4, stride=2, padding=1)
		### 32 x 32 x 128
		self.upsample5			= nn.ConvTranspose2d(self.size * 8 , self.size * 8 , kernel_size=3, stride=1, padding=1)
		### 32 x 32 x 128
		self.upsample6			= nn.ConvTranspose2d(self.size * 8 , self.size * 8 , kernel_size=3, stride=1, padding=1)
		### 32 x 32 x 128
		self.upsample7			= nn.ConvTranspose2d(self.size * 8 , self.size * 4 , kernel_size=4, stride=2, padding=1)
		### 64 x 64 x 64
		self.upsample8			= nn.ConvTranspose2d(self.size * 4 , self.size * 4 , kernel_size=3, stride=1, padding=1)
		### 64 x 64 x 64
		self.upsample9			= nn.ConvTranspose2d(self.size * 4 , self.size * 4 , kernel_size=3, stride=1, padding=1)
		### 64 x 64 x 64
		self.upsample10			= nn.ConvTranspose2d(self.size * 4 , self.size * 2 , kernel_size=4, stride=2, padding=1)
		### 128 x 128 x 32
		self.upsample11			= nn.ConvTranspose2d(self.size * 2 , self.size * 2 , kernel_size=3, stride=1, padding=1)
		### 128 x 128 x 32
		self.upsample12			= nn.ConvTranspose2d(self.size * 2 , self.size	   , kernel_size=4, stride=2, padding=1)
		### 256 x 256 x 16
		self.upsample13			= nn.ConvTranspose2d(self.size	   , self.size     , kernel_size=3, stride=1, padding=1)
		### 256 x 256 x 16
		self.upsample14			= nn.ConvTranspose2d(self.size	   , self.channel  , kernel_size=3, stride=1, padding=1)
		### 256 x 256 x 3
		self.upsample15			= nn.ConvTranspose2d(self.channel  , self.channel  , kernel_size=3, stride=1, padding=1)
		### 256 x 256 x 3
		self.upsample16			= nn.ConvTranspose2d(self.channel  , self.channel  , kernel_size=3, stride=1, padding=1)
		### 256 x 256 x 3

		self.sigmoid			= nn.Sigmoid()

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
		se = self.blick9(se)
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

		pos= self.sigmoid(pos)
		return pos

class Bottleneck(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		
	def forward(self, x):
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class ResFCN120(nn.Module):
	def __init__(self, resolution_input=120, resolution_output=120, channel=3, size=64):
		super().__init__()
		self.input_resolution 	= resolution_input
		self.output_resolution	= resolution_output
		self.channel			= channel
		self.size				= size
		
		self.inplanes = 64
		self.dilation = 1

		### 120 x 120 x 3
		self.conv1				= nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		### 60 x 60 x 64
		self.bn1 				= nn.BatchNorm2d(self.inplanes)
		### 60 x 60 x 64
		self.relu 				= nn.ReLU(inplace=True)
		### 60 x 60 x 64
		self.maxpool 			= nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		### 30 x 30 x 64
		self.layer1				= self._make_encode_layer(64 , 3)
		### 30 x 30 x 256
		self.layer2				= self._make_encode_layer(128, 4, stride=2)
		### 15 x 15 x 512
		self.layer3				= self._make_encode_layer(256, 6, stride=2)
		###	8 x 8 x 1024
		self.layer4				= self._make_encode_layer(512, 3, stride=2)
		###	4 x 4 x 2048
		self.upsample0			= nn.ConvTranspose2d(self.size * 32, self.size * 32, kernel_size=3, stride=1, padding=1)
		### 4 x 4 x 2048
		self.upsample1			= nn.ConvTranspose2d(self.size * 32, self.size * 16, kernel_size=4, stride=2, padding=1)
		### 8 x 8 x 1024
		self.upsample2			= nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
		### 8 x 8 x 1024
		self.upsample3			= nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
		### 8 x 8 x 1024
		self.upsample4			= nn.ConvTranspose2d(self.size * 16, self.size * 8 , kernel_size=4, stride=2, padding=1)
		### 15 x 15 x 512
		self.upsample5			= nn.ConvTranspose2d(self.size * 8 , self.size * 8 , kernel_size=3, stride=1, padding=1)
		### 15 x 15 x 512
		self.upsample6			= nn.ConvTranspose2d(self.size * 8 , self.size * 8 , kernel_size=3, stride=1, padding=1)
		### 15 x 15 x 512
		self.upsample7			= nn.ConvTranspose2d(self.size * 8 , self.size * 4 , kernel_size=4, stride=2, padding=1)
		### 30 x 30 x 256
		self.upsample8			= nn.ConvTranspose2d(self.size * 4 , self.size * 4 , kernel_size=3, stride=1, padding=1)
		### 30 x 30 x 256
		self.upsample9			= nn.ConvTranspose2d(self.size * 4 , self.size * 4 , kernel_size=3, stride=1, padding=1)
		### 30 x 30 x 256
		self.upsample10			= nn.ConvTranspose2d(self.size * 4 , self.size * 2 , kernel_size=4, stride=2, padding=1)
		### 60 x 60 x 128
		self.upsample11			= nn.ConvTranspose2d(self.size * 2 , self.size * 2 , kernel_size=3, stride=1, padding=1)
		### 60 x 60 x 128
		self.upsample12			= nn.ConvTranspose2d(self.size * 2 , self.size	   , kernel_size=4, stride=2, padding=1)
		### 120 x 120 x 64
		self.upsample13			= nn.ConvTranspose2d(self.size	   , self.size     , kernel_size=3, stride=1, padding=1)
		### 120 x 120 x 64
		self.upsample14			= nn.ConvTranspose2d(self.size	   , self.channel  , kernel_size=3, stride=1, padding=1)
		### 120 x 120 x 3
		self.upsample15			= nn.ConvTranspose2d(self.channel  , self.channel  , kernel_size=3, stride=1, padding=1)
		### 120 x 120 x 3
		self.upsample16			= nn.ConvTranspose2d(self.channel  , self.channel  , kernel_size=3, stride=1, padding=1)
		### 120 x 120 x 3

	def _make_encode_layer(self, planes, blocks, stride=1):
		if stride != 1 or self.inplanes != planes * 4:
			downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, stride), nn.BatchNorm2d(planes * 4),)
		layers = []
		layers.append(Bottleneck(self.inplanes, planes, stride, downsample, 1, 64, 1))
		self.inplanes = planes * 4
		for _ in range(1, blocks):
			layers.append(Bottleneck(self.inplanes, planes, 1, 64, 1))

		return nn.Sequential(*layers)

	def _make_decode_layer(self, planes, blocks, stride=1):
		if stride != 1 or self.inplanes != planes * 4:
			upsample = nn.Sequential(nn.ConvTranspose2d(self.inplanes, planes * 4, stride), nn.BatchNorm2d(planes * 4),)
		layers = []
		layers.append(Bottleneck(self.inplanes, planes, stride, upsample, 1, 64, 1))
		self.inplanes = planes * 4
		for _ in range(1, blocks):
			layers.append(Bottleneck(self.inplanes, planes, 1, 64, 1))
		return nn.Sequential(*layers)
	def forward(self, x):
		se = self.conv1(x)
		se = self.bn1(x)
		se = self.relu(x)
		se = self.maxpool(x)
		se = self.layer1(se)
		se = self.layer2(se)
		se = self.layer3(se)
		se = self.layer4(se)

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
		
		return pos

