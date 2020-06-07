''' 
Generate uv position map of 300W_LP.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
from time import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle
import cv2
from params import *

import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel


working_folder  = "/home/viet/Projects/Pycharm/SPRNet/"
FLAGS = {   
            "data_path"     : "/home/viet/Projects/Pycharm/SPRNet/data/test.data/AFLW2000-3D/",
			"list_path"     : os.path.join(working_folder, "data/test.data/AFLW2000-3D.list"),
            "param_path"    : os.path.join(working_folder, "train.configs/param_all_norm.pkl"),
			"save_uv_path"  : os.path.join(working_folder, "data/verify_uv_256x256"),
			"save_im_path"  : os.path.join(working_folder, "data/verify_im_256x256"),
			"bfm_path"      : os.path.join(working_folder, "data/processing/Data/BFM/Out/BFM.mat"),
            "bfm_uv_path"   : os.path.join(working_folder, "data/processing/Data/BFM/Out/BFM_UV.mat"),
            "uv_h"          : 256,
			"uv_w"          : 256,
			"image_h"       : 256,
			"image_w"       : 256,
            "num_worker"    : 8,
            "is62Param"     : False
		}

uv_kpt_ind  = np.loadtxt(os.path.join(working_folder, "data/processing/Data/UV/uv_kpt_ind.txt")).astype(np.int32)
face_ind    = np.loadtxt(os.path.join(working_folder, "data/processing/Data/UV/face_ind.txt")).astype(np.int32)
triangles   = np.loadtxt(os.path.join(working_folder, "data/processing/Data/UV/triangles.txt")).astype(np.int32)

def reconstruct_vertex(param, whitening=True, dense=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp"""
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    if dense:
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
    else:
        """For 68 pts"""
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset
        # for landmarks
        vertex[1, :] = std_size + 1 - vertex[1, :]
    return vertex

def process_uv(uv_coords, uv_h = 256, uv_w = 120):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def show_lb_mesh(img_path, vertices, keypoint_index):
    img = cv2.imread(img_path)

    zoom  =  2

    img = cv2.resize(img, None, fx=32/15 * zoom,fy=32/15 * zoom,interpolation = cv2.INTER_CUBIC)

    x, y, z = vertices * zoom
    for i in range(0, x.shape[0], 1):
        if i in keypoint_index:
            img = cv2.circle(img, (int(x[i]),256 * zoom - int(y[i])), 4, (255, 255, 255), -1)
        else:
            img = cv2.circle(img, (int(x[i]),256 * zoom - int(y[i])), 1, (255, 0, int(z[i])), -1)
    cv2.imshow("lb_point_scatter",img)
    cv2.waitKey()

def show_ALFW_mesh(nimg, vertices, keypoint_index):
    img = (nimg * 255.0).astype(np.uint8)

    x, y, z = vertices
    for i in range(0, x.shape[0], 1):
        if i in keypoint_index:
            img = cv2.circle(img, (int(x[i]),int(y[i])), 4, (255, 255, 255), -1)
        # else:
        #     img = cv2.circle(img, (int(x[i]),int(y[i])), 1, (255, 0, int(z[i])), -1)
    cv2.imshow("lb_point_scatter",img)
    cv2.waitKey()

def show_lb_mesh(img_path, vertices, keypoint_index):
    img = cv2.imread(img_path)

    zoom  =  2

    img = cv2.resize(img, None, fx=32/15 * zoom,fy=32/15 * zoom,interpolation = cv2.INTER_CUBIC)

    x, y, z = vertices * zoom
    for i in range(0, x.shape[0], 1):
        if i in keypoint_index:
            img = cv2.circle(img, (int(x[i]),256 * zoom - int(y[i])), 4, (255, 255, 255), -1)
        else:
            img = cv2.circle(img, (int(x[i]),256 * zoom - int(y[i])), 1, (255, 0, int(z[i])), -1)
    cv2.imshow("lb_point_scatter",img)
    cv2.waitKey()

def show_uv_mesh(img_path, uv, keypoint):
    img = cv2.imread(img_path)
    [h, w, c] = img.shape
    # H W 3
    # 3 (W*H)
    zoom  =  2

    x, y, z = uv.transpose(2, 0, 1).reshape(3, -1) * zoom
    img = cv2.resize(img, None, fx=256/h * zoom, fy=256/w * zoom,interpolation = cv2.INTER_CUBIC)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (255, 0, int(z[i])), -1)
    x, y, z = keypoint.transpose().astype(np.int32) * zoom
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 4, (255, 255, 255), -1)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()

def generate_posmap_lb_62params(bfm, image_path, param, save_uv_folder, save_img_folder, uv_h = 256, uv_w = 256, image_h = 256, image_w = 256):
    ### 1. load image and resize from 120 to 256
    image_name = image_path.strip().split('/')[-1]
    # img = io.imread(image_path)
    # img = cv2.resize(img, None, fx=32/15,fy=32/15, interpolation = cv2.INTER_CUBIC)
    # image = img/255
    # [h, w, c] = image.shape

    ### 2. reconstruct vertex from 62 BFM parameters
    vertices = reconstruct_vertex(param, dense = True).astype(np.float32) * 32 / 15
    projected_vertices = vertices.transpose()
    image_vertices = projected_vertices.copy()
    image_vertices[:,1] = image_h - image_vertices[:,1]

    position = image_vertices.copy()

    ### 3. render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c = 3)

    ### 4. get 68 key points index ~> visualize
    # kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    # show_lb_mesh(image_path, vertices, bfm.kpt_ind)
    # kpt = uv_position_map[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    # show_uv_mesh(image_path, uv_position_map, kpt)
    ### 5. save files
    # io.imsave('{}/{}'.format(save_folder, image_name), np.squeeze(cropped_image))
    np.save('{}/{}'.format(save_uv_folder, image_name.replace('jpg', 'npy')), uv_position_map)
    io.imsave('{}/{}'.format(save_img_folder, image_name), img)

def generate_posmap_lb_fitting(bfm, image_path, mat_path, save_uv_folder, save_img_folder, uv_h = 256, uv_w = 256, image_h = 256, image_w = 256):
    ### 1. load image and fitted parameters
    image_name  = image_path.strip().split('/')[-1]
    image       = io.imread(image_path)/255.
    [h, w, c] = image.shape

    info = sio.loadmat(mat_path)
    pose_para = info['Pose_Para'].T.astype(np.float32)
    shape_para = info['Shape_Para'].astype(np.float32)
    exp_para = info['Exp_Para'].astype(np.float32)

    ### 2. generate mesh
    # generate shape
    vertices = bfm.generate_vertices(shape_para, exp_para)
    # transform mesh
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]
    transformed_vertices    = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices      = transformed_vertices.copy()
    image_vertices          = projected_vertices.copy()
    image_vertices[:,1]     = h - image_vertices[:,1]

    ### 3. crop square image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top)/2
    size = int(old_size*1.5)
    
    # random the img margin
    marg = old_size*0.1
    t_x = np.random.rand()*marg*2 - marg
    t_y = np.random.rand()*marg*2 - marg
    center[0] = center[0]+t_x
    center[1] = center[1]+t_y
    size = size*(np.random.rand()*0.2 + 0.9)

    # crop and record the transform parameters
    [crop_h, crop_w, crop_c] = [image_h, image_w, 3]
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], 
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0          ,          0], 
                        [0          , crop_h - 1], 
                        [crop_w - 1 ,          0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, dst_pts)
    trans_mat = tform.params
    trans_mat_inv = tform._inv_matrix
    scale = trans_mat[0][0]
    cropped_image = skimage.transform.warp(image, trans_mat_inv, output_shape=(crop_h, crop_w))
    # transform face position(image vertices) along with 2d facial image
    position = image_vertices.copy()
    position[:, 2]  = 1
    position        = np.dot(position, trans_mat.T)
    position[:, 2]  = image_vertices[:, 2]*scale # scale z
    position[:, 2]  = position[:, 2] - np.min(position[:, 2]) # translate z

    ### 4. render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c = 3)
    ### 5. get 68 key points index ~> visualize
    # kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    # show_ALFW_mesh(cropped_image, position.transpose(), bfm.kpt_ind)
    # kpt = uv_position_map[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    # show_uv_mesh(os.path.join(save_img_folder, image_name), uv_position_map, kpt)
    ### 6. save files
    io.imsave('{}/{}'.format(save_img_folder, image_name), (np.squeeze(cropped_image * 255.0)).astype(np.uint8))
    np.save('{}/{}'.format(save_uv_folder, image_name.replace('jpg', 'npy')), uv_position_map)

if __name__ == '__main__':
    if not os.path.exists(FLAGS["save_uv_path"]):
        os.mkdir(FLAGS["save_uv_path"])

    if not os.path.exists(FLAGS["save_im_path"]):
        os.mkdir(FLAGS["save_im_path"])

    # load uv coords
    global uv_coords
    uv_coords = face3d.morphable_model.load.load_uv_coords(FLAGS["bfm_uv_path"]) 
    uv_coords = process_uv(uv_coords, FLAGS["uv_h"], FLAGS["uv_w"])
    
    # load bfm 
    bfm = MorphabelModel(FLAGS["bfm_path"]) 
    img_names_list          = Path(FLAGS["list_path"]).read_text().strip().split('\n')
    if FLAGS["is62Param"]:
        param_62d               = pickle.load(open(FLAGS["param_path"],'rb'))
        index = 0
        for img_name in tqdm(img_names_list):
            file_name   = os.path.splitext(img_name)[0]
            image_path  = os.path.join(FLAGS["data_path"], file_name + ".jpg")
            param       = param_62d[index]
            generate_posmap_lb_62params(bfm, image_path, param, FLAGS["save_uv_path"], FLAGS["save_im_path"])
            index       = index + 1
    else:
        for img_name in tqdm(img_names_list):
            file_name   = os.path.splitext(img_name)[0]
            image_path  = os.path.join(FLAGS["data_path"], file_name + ".jpg")
            mat_path    = os.path.join(FLAGS["data_path"], file_name + ".mat")
            generate_posmap_lb_fitting(bfm, image_path, mat_path, FLAGS["save_uv_path"], FLAGS["save_im_path"])
