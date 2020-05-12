import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import cv2

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

def preprocess(mask):
    """
    :param mask: grayscale of mask.
    :return:
    """
    tmp = {}
    mask[mask > 0] = mask[mask > 0] / 16
    mask[mask == 15] = 16
    mask[mask == 7] = 8
    
    return mask

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _fspecial_gauss(window_size, sigma=1.5):
    # Function to mimic the 'fspecial' gaussian MATLAB function.
    coords = np.arange(0, window_size, dtype=np.float32)
    coords -= (window_size - 1) / 2.0

    g = coords ** 2
    g *= (-0.5 / (sigma ** 2))
    g = np.reshape(g, (1, -1)) + np.reshape(g, (-1, 1))
    g = torch.from_numpy(np.reshape(g, (1, -1)))
    g = torch.softmax(g, dim=1)
    g = g / g.sum()
    return g


# 2019.05.26. butterworth filter.
# ref: http://www.cnblogs.com/laumians-notes/p/8592968.html
def butterworth(window_size, sigma=1.5, n=2):
    nn = 2 * n
    bw = torch.Tensor([1 / (1 + ((x - window_size // 2) / sigma) ** nn) for x in range(window_size)])
    return bw / bw.sum()


def create_window(window_size, channel=3, sigma=1.5, gauss='original', n=2):
    if gauss == 'original':
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    elif gauss == 'butterworth':
        _1D_window = butterworth(window_size, sigma, n).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    else:
        g = _fspecial_gauss(window_size, sigma)
        g = torch.reshape(g, (1, 1, window_size, window_size))
        # 2019.06.05.
        # https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853
        g = tile(g, 0, 3)
        return g


def _ssim(img1, img2, window_size=11, window=None, val_range=2, size_average=True):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    # padd = window_size//2
    padd = 0
    (batch, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # 2019.05.05
    # pytorch默认是NCHW. 跟caffe一样.
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_square = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_square = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12_square = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12_square + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_square + sigma2_square + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

### SSIM Loss
class ORIGINAL_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, val_range=2, size_average=True):
        super(ORIGINAL_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - _ssim(img1, img2, self.window_size, window, self.val_range, self.size_average)


def dfl_ssim(img1, img2, mask, window_size=11, val_range=1, gauss='original'):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    # padd = window_size//2
    padd = 0
    (batch, channel, height, width) = img1.size()
    img1, img2 = torch.mul(img1, mask), torch.mul(img2, mask)

    real_size = min(window_size, height, width)
    window = create_window(real_size, gauss=gauss).to(img1.device)

    # 2019.05.07.
    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    num0 = mu1 * mu2 * 2.0
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    den0 = mu1_sq + mu2_sq

    luminance = (num0 + c1) / (den0 + c1)

    num1 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) * 2.0
    den1 = F.conv2d(img1 * img1 + img2 * img2, window, padding=padd, groups=channel)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)
    ssim_val = torch.mean(luminance * cs, dim=(-3, -2))

    return torch.mean((1.0 - ssim_val) / 2.0)

### Weight Mask Loss
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

    def forward(self, output_l, target_l): #output_l, output_r, label, target_l, target_r):
        result_l = torch.mean(torch.pow((output_l - target_l), 2), dim=1)
        result_l = torch.mul(result_l, self.mask)

        result_l = torch.sum(result_l)
        result_l = result_l / (self.mask.size(1) ** 2)


        result_r = torch.mean(torch.pow((output_r - target_r), 2), dim=1)
        result_r = torch.mul(result_r, self.mask)

        result_r = torch.sum(result_r)
        result_r = result_r / (self.mask.size(1) ** 2)
        # result = torch.mean(result)
        is_right = np.random.choice([0, 1], p=[0.5, 0.5])
        return result_l if is_right else result_r

### Shape Loss
class UVLoss0(nn.Module):
    """
        
    """

    def __init__(self, mask_path, window_size=11, alpha=0.8, gauss='original'):
        super(UVLoss0, self).__init__()
        self.window_size = window_size
        self.window      = None
        self.channel     = None

        self.gauss       = gauss
        self.alpha       = alpha

        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, 0)
            self.mask = torch.from_numpy(preprocess(self.mask)).float().to("cuda")
        else:
            raise FileNotFoundError("Mask File Not Found! Please Check your Settings!")

    def forward(self, output_l, output_r, label, target_l, target_r):
        (_, channel, _, _) = output_l.size()
        self.channel = channel
        
        loss_ssim_1 = 10 * dfl_ssim(output_l, target_l, mask=self.mask, window_size=self.window_size, gauss=self.gauss)
        loss_ssim_2 = 10 * dfl_ssim(output_r, target_r, mask=self.mask, window_size=self.window_size, gauss=self.gauss)
        # result = torch.mean(result)
        return loss_ssim_1 + loss_ssim_2

### Constrain Identification Loss
