import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import cv2

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

class WeightMaskLoss(nn.Module):
    """
        L2_Loss * Weight Mask
    """

    def __init__(self, mask_path):
        super(WeightMaskLoss, self).__init__()
        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, 0)
            self.mask = torch.from_numpy(preprocess(self.mask)).float().to("cuda")
        else:
            raise FileNotFoundError("Mask File Not Found! Please Check your Settings!")

    def forward(self, pred, gt):
        result = torch.mean(torch.pow((pred - gt), 2), dim=1)
        result = torch.mul(result, self.mask)

        result = torch.sum(result)
        result = result / (self.mask.size(1) ** 2)
        
        # result = torch.mean(result)
        return result

