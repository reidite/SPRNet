import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.params import *
from math import sqrt
from torch.autograd import Variable
from models.io_utils import _load, _numpy_to_cuda, _numpy_to_tensor, _load_gpu

_to_tensor = _numpy_to_cuda

def _parse_param_batch(param):
	N 			= 	param.shape[0]
	p_			= 	param[:, :, :12].view(N, 3, -1)
	p			= 	p_[:, :, :3]
	offset		= 	p_[:, :, -1].view(N, 3, 1)
	alpha_shp	=	param[:, 12:52].view(N, -1, 1)
	alpha_exp	=	param[:, 52:].view(N, -1, 1)
	return p, offset, alpha_shp, alpha_exp

class WPDCLoss_0(nn.Module):
