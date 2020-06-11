import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
from models.resfcn import load_SPRNET
from data.dataset import USRTestDataset, ToTensor
from torch.utils.data import DataLoader
from math import cos, sin, atan2, asin

working_folder  = "/home/viet/Projects/Pycharm/SPRNet/"
FLAGS = {   
            "model"             : os.path.join(working_folder, "train_log/_checkpoint_epoch_80.pth.tar"),
            "data_path"         : os.path.join(working_folder, "data"),
			"uv_kpt_ind_path"   : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "face_ind_path"     : os.path.join(working_folder, "data/processing/Data/UV/face_ind.txt"),
            "triangles_path"    : os.path.join(working_folder, "data/processing/Data/UV/triangles.txt"),
            "canonical_vts_path": os.path.join(working_folder, "data/processing/Data/UV/canonical_vertices.npy"),
			"result_path"       : os.path.join(working_folder, "result/usr"),
            "uv_kpt_path"       : os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt"),
            "device"            : "cuda",
            "devices_id"        : [0],
            "batch_size"        : 16, 
            "workers"           : 8
		}

uv_kpt_ind          = np.loadtxt(FLAGS["uv_kpt_ind_path"]).astype(np.int32)
face_ind            = np.loadtxt(FLAGS["face_ind_path"]).astype(np.int32)
triangles           = np.loadtxt(FLAGS["triangles_path"]).astype(np.int32)
canonical_vertices  = np.load(FLAGS["canonical_vts_path"])
resolution  = 256

def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z

def P2sRt(P):
    ''' decomposing camera matrix P. 
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation. 
    '''
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d

def compute_similarity_transform(points_static, points_to_transform):
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3, 1)
    t1 = -np.mean(p1, axis=1).reshape(3, 1)
    t_final = t1 - t0

    p0c = p0 + t0
    p1c = p1 + t1

    covariance_matrix = p0c.dot(p1c.T)
    U, S, V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0) ** 2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0) ** 2))

    s = (rms_d0 / rms_d1)
    P = np.c_[s * np.eye(3).dot(R), t_final]
    return P

def estimate_pose(vertices):
    P = compute_similarity_transform(vertices, canonical_vertices)
    s, R, t = P2sRt(P)
    pose = matrix2angle(R)

    return P, pose, (s, R, t)

def frontalize(vertices):
    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)

    return front_vertices

def get_landmarks(pos):
    kpt = pos[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    return kpt

def get_vertices(pos):
    all_vertices = np.reshape(pos, [resolution**2, -1])
    vertices = all_vertices[face_ind, :]
    return vertices

def get_colors_from_texture(texture):
    all_colors = np.reshape(texture, [resolution**2, -1])
    colors = all_colors[face_ind, :]
    return colors

def get_colors(image, vertices):
    [h, w, _] = image.shape
    vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)
    vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:,1], ind[:,0], :]
    return colors

def show_uv_mesh(img, uv, keypoint, isMesh=True):
    img = cv2.resize(img, (256,256))
    img = cv2.resize(img, None, fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    if isMesh:
        x, y, z = uv.transpose(2, 0, 1).reshape(3, -1) * 2
        for i in range(0, x.shape[0], 1):
            img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (255, 0, 0), -1)
    x, y, z = keypoint.transpose().astype(np.int32) * 2
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 4, (255, 255, 255), -1)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# def predict():
#     model               = load_SPRNET()
#     torch.cuda.set_device(FLAGS["devices_id"][0])
#     model	            = nn.DataParallel(model, device_ids=FLAGS["devices_id"]).cuda()
#     pretrained_weights  = torch.load(FLAGS["model"], map_location='cuda:0')
#     model.load_state_dict(pretrained_weights['state_dict'])
#     dataset 	        = USRTestDataset(
# 		root_dir  		= FLAGS["data_path"],
# 		transform 		= transforms.Compose([ToTensor()])
# 	)

#     data_loader = DataLoader(
# 		dataset, 
# 		batch_size      =   FLAGS["batch_size"], 
# 		num_workers     =   FLAGS["workers"],
# 		shuffle         =   False, 
# 		pin_memory      =   True, 
# 		drop_last       =   True
# 	)

#     cudnn.benchmark = True   
#     model.eval()
#     with torch.no_grad():
#         for i, (img) in tqdm(enumerate(data_loader)):
#             gens  = model(img, isTrain=False)[:, :, 1:257, 1:257]