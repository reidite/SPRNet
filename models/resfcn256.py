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
	def __init__(self, resolution_input=256, resolution_output=256, channel=3, size=16):
		super().__init__()
		self.input_resolution 	= resolution_input
		self.output_resolution	= resolution_output
		self.channel			= channel
		self.size				= size

		self.block0				= conv3x3(in_planes=3, out_planes=self.size, padding="same")
		self.block1				= ResBlock(inplanes=self.size, planes=self.size * 2, stride=2)
		self.block2				= ResBlock(inplanes=self.size * 2, planes=self.size * 2, stride=1)
		self.block3				= ResBlock(inplanes=self.size * 2, planes=self.size * 4, stride=2)
		self.block4				= ResBlock(inplanes=self.size * 4, planes=self.size * 4, stride=1)
		self.block5				= ResBlock(inplanes=self.size * 4, planes=self.size * 8, stride=2)
		self.block6				= ResBlock(inplanes=self.size * 8, planes=self.size * 8, stride=1)
		self.block7				= ResBlock(inplanes=self.size * 8, planes=self.size * 16, stride=2)
		self.block8				= ResBlock(inplanes=self.size * 16, planes=self.size * 16,stride=1)
		self.block9				= ResBlock(inplanes=self.size * 16, planes=self.size * 32, stride=2)
		self.block10			= ResBlock(inplanes=self.size * 32, planes=self.size * 32, stride=1)

		self.upsample0			= nn.ConvTranspose2d(self.size * 32, self.size * 32, kernel_size=3, stride=1, padding=1)
		self.upsample1			= nn.ConvTranspose2d(self.size * 32, self.size * 16, kernel_size=4, stride=2, padding=1)
		self.upsample2			= nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
		self.upsample3			= nn.ConvTranspose2d(self.size * 16, self.size * 16, kernel_size=3, stride=1, padding=1)
		self.upsample4			= nn.ConvTranspose2d(self.size * 16, self.size * 8 , kernel_size=4, stride=2, padding=1)
		self.upsample5			= nn.ConvTranspose2d(self.size * 8 , self.size * 8 , kernel_size=3, stride=1, padding=1)
		self.upsample6			= nn.ConvTranspose2d(self.size * 8 , self.size * 8 , kernel_size=3, stride=1, padding=1)
		self.upsample7			= nn.ConvTranspose2d(self.size * 8 , self.size * 4 , kernel_size=4, stride=2, padding=1)
		self.upsample8			= nn.ConvTranspose2d(self.size * 4 , self.size * 4 , kernel_size=3, stride=1, padding=1)
		self.upsample9			= nn.ConvTranspose2d(self.size * 4 , self.size * 4 , kernel_size=3, stride=1, padding=1)
		self.upsample10			= nn.ConvTranspose2d(self.size * 4 , self.size * 2 , kernel_size=4, stride=2, padding=1)
		self.upsample11			= nn.ConvTranspose2d(self.size * 2 , self.size * 2 , kernel_size=3, stride=1, padding=1)
		self.upsample12			= nn.ConvTranspose2d(self.size * 2 , self.size	   , kernel_size=4, stride=2, padding=1)
		self.upsample13			= nn.ConvTranspose2d(self.size	   , self.size     , kernel_size=3, stride=1, padding=1)
		self.upsample14			= nn.ConvTranspose2d(self.size	   , self.channel  , kernel_size=3, stride=1, padding=1)
		self.upsample15			= nn.ConvTranspose2d(self.channel  , self.channel  , kernel_size=3, stride=1, padding=1)
		self.upsample16			= nn.ConvTranspose2d(self.channel  , self.channel  , kernel_size=3, stride=1, padding=1)

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

