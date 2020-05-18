import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io, transform
import math
import os
import cv2


from math import sqrt
from torch.autograd import Variable
from models.io_utils import _load, _numpy_to_cuda, _numpy_to_tensor, _load_gpu

_to_tensor = _numpy_to_cuda

#region UV Loss
def UVLoss(is_foreface=False, is_weighted=False, is_nme=False):
#endregion

#region UV Loss
def PRNError(is_2d=False, is_normalized=True, is_foreface=True, is_landmark=False, is_gt_landmark=False):
    def templateError(y_gt, y_fit, bbox=None, landmarks=None):
        assert (not (is_foreface and is_landmark))
        y_true = y_gt.copy()
        y_pred = y_fit.copy()
        y_true[:, :, 2] = y_true[:, :, 2] * face_mask_np
        y_pred[:, :, 2] = y_pred[:, :, 2] * face_mask_np
        y_true_mean = np.mean(y_true[:, :, 2]) * face_mask_mean_fix_rate
        y_pred_mean = np.mean(y_pred[:, :, 2]) * face_mask_mean_fix_rate
        y_true[:, :, 2] = y_true[:, :, 2] - y_true_mean
        y_pred[:, :, 2] = y_pred[:, :, 2] - y_pred_mean

        if is_landmark:
            # the gt landmark is not the same as the landmarks get from mesh using index
            if is_gt_landmark:
                gt = landmarks.copy()
                gt[:, 2] = gt[:, 2] - gt[:, 2].mean()

                pred = y_pred[uv_kpt[:, 0], uv_kpt[:, 1]]
                diff = np.square(gt - pred)
                if is_2d:
                    dist = np.sqrt(np.sum(diff[:, 0:2], axis=-1))
                else:
                    dist = np.sqrt(np.sum(diff, axis=-1))
            else:
                gt = y_true[uv_kpt[:, 0], uv_kpt[:, 1]]
                pred = y_pred[uv_kpt[:, 0], uv_kpt[:, 1]]
                gt[:, 2] = gt[:, 2] - gt[:, 2].mean()
                pred[:, 2] = pred[:, 2] - pred[:, 2].mean()
                diff = np.square(gt - pred)
                if is_2d:
                    dist = np.sqrt(np.sum(diff[:, 0:2], axis=-1))
                else:
                    dist = np.sqrt(np.sum(diff, axis=-1))
        else:
            diff = np.square(y_true - y_pred)
            if is_2d:
                dist = np.sqrt(np.sum(diff[:, :, 0:2], axis=-1))
            else:
                # 3d
                dist = np.sqrt(np.sum(diff, axis=-1))
            if is_foreface:
                dist = dist * face_mask_np * face_mask_mean_fix_rate

        if is_normalized:  # 2D bbox size
            # bbox_size = np.sqrt(np.sum(np.square(bbox[0, :] - bbox[1, :])))
            if is_landmark:
                bbox_size = np.sqrt((bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1]))
            else:
                face_vertices = y_gt[face_mask_np > 0]
                minx, maxx = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
                miny, maxy = np.min(face_vertices[:, 1]), np.max(face_vertices[:, 1])
                llength = np.sqrt((maxx - minx) * (maxy - miny))
                bbox_size = llength
        else:  # 3D bbox size
            face_vertices = y_gt[face_mask_np > 0]
            minx, maxx = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
            miny, maxy = np.min(face_vertices[:, 1]), np.max(face_vertices[:, 1])
            minz, maxz = np.min(face_vertices[:, 2]), np.max(face_vertices[:, 2])
            if is_landmark:
                llength = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
            else:
                llength = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2 + (maxz - minz) ** 2)
            bbox_size = llength

        loss = np.mean(dist / bbox_size)
        return loss

    return templateError
def getErrorFunction(error_func_name='NME'):
    if error_func_name == 'nme2d' or error_func_name == 'normalized mean error2d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=True)
    elif error_func_name == 'nme3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=True)
    elif error_func_name == 'landmark2d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True)
    elif error_func_name == 'landmark3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True)
    elif error_func_name == 'gtlandmark2d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True)
    elif error_func_name == 'gtlandmark3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True)
    else:
        print('unknown error:', error_func_name)
#endregion
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
