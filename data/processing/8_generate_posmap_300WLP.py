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

uv_kpt_ind = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/uv_kpt_ind.txt").astype(np.int32)
face_ind = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/face_ind.txt").astype(np.int32)
triangles = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/triangles.txt").astype(np.int32)

def reconstruct_vertex(param, whitening=True, dense=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp"""
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))         #qian 12 dian ying gai shi u_base,concatenate() shi xiang pin jie yi ge 62 wei de shu zu
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean       #   x-mean/std ~ N(0,1)   you he yong ?????
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]        #suo you hang,qian 3 lie
    offset = p_[:, -1].reshape(3, 1)     #suo you hang ,dao shu di yi lie, tai sao le,bu neng ren!!!
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    # Note !
    # dense and non-dense difference main display :w_shp:159645x40; w_shp:204x40
    if dense:      
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
    else:                                                          # reshape(3, -1, order='F') order='F'竖着读，竖着写，优先读/写一列
        """For 68 pts"""  # get 68 keypoint 3d position  p:3x3 (u + w_shp...):159645x1--->3x53215
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

def show_mesh(img_path, vertices, keypoint_index):
    img = cv2.imread(img_path)

    zoom  =  2

    img = cv2.resize(img, None, fx=32/15 * zoom,fy=32/15 * zoom,interpolation = cv2.INTER_CUBIC)

    cv2.imshow("origin",img)
    x, y, z = vertices * zoom
    for i in range(0, x.shape[0], 1):
        if i in keypoint_index:
            img = cv2.circle(img, (int(x[i]),256 * zoom - int(y[i])), 4, (255, 255, 255), -1)
        else:
            img = cv2.circle(img, (int(x[i]),256 * zoom - int(y[i])), 1, (255, 0, 130 - z[i]), -1)
    cv2.imshow("3d_point_scatter",img)
    cv2.waitKey()

def show_uv_mesh(img_path, uv, keypoint):
    img = cv2.imread(img_path)
    # H W 3
    # 3 (W*H)
    zoom  =  2

    x, y, z = uv.transpose(2, 0, 1).reshape(3, -1) * zoom
    img = cv2.resize(img, None, fx=32/15 * zoom, fy=32/15 * zoom,interpolation = cv2.INTER_CUBIC)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (255, 0, 0), -1)
    x, y, z = keypoint.transpose().astype(np.int32) * zoom
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 4, (255, 255, 255), -1)
    # res = cv2.resize(img, None, fx=3,fy=3,interpolation = cv2.INTER_CUBIC)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()

def run_posmap_300W_LP(bfm, image_path, param, save_uv_folder, save_img_folder, uv_h = 256, uv_w = 256, image_h = 256, image_w = 256):
    # 1. load image and fitted parameters
    image_name = image_path.strip().split('/')[-1]
    img = io.imread(image_path)
    img = cv2.resize(img, None, fx=32/15,fy=32/15, interpolation = cv2.INTER_CUBIC)
    print(img.shape)
    image = img/255
    [h, w, c] = image.shape

    # pose_para = info['Pose_Para'].T.astype(np.float32)
    # shape_para = info['Shape_Para'].astype(np.float32)
    # exp_para = info['Exp_Para'].astype(np.float32)

    #   2. generate mesh
    # generate shape
    # vertices = bfm.generatemesh_vertices(shape_para, exp_para)
    #    transform 
    # s = pose_para[-1, 0]
    # angles = pose_para[:3, 0]
    # t = pose_para[3:6, 0]
    # transformed_vertices  = bfm.transform_3ddfa(vertices, s, angles, t)
    # projected_vertices    = transformed_vertices.copy() # using stantard camera & orth projection as in 3DDFA

    vertices = reconstruct_vertex(param, dense = True).astype(np.float32) * 32 / 15
    projected_vertices = vertices.transpose()
    image_vertices = projected_vertices.copy()
    image_vertices[:,1] = h - image_vertices[:,1] - 1
    
    # 3. crop image with key points
    # image_vertices [53215, 3]
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    # show_mesh(image_path, vertices, bfm.kpt_ind)
    # kpt [68: 3]
    left = np.min(kpt[:, 0])
    # min x = float32
    right = np.max(kpt[:, 0])
    # max x = float32
    top = np.min(kpt[:, 1])
    # min y = float32
    bottom = np.max(kpt[:, 1])
    # max y = float32
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    # center = [2,]
    old_size = (right - left + bottom - top)/2
    # old_size = float
    size = int(old_size*1.5)
    # random pertube. you can change the numbers
    marg = old_size*0.1
    t_x = np.random.rand()*marg*2 - marg
    t_y = np.random.rand()*marg*2 - marg
    center[0] = center[0]+t_x; center[1] = center[1]+t_y
    size = size*(np.random.rand()*0.2 + 0.9)

    # crop and record the transform parameters
    # src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    # DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    # tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    # cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))


    # transform face position(image vertices) along with 2d facial image 
    position = image_vertices.copy()
    position[:, 2] = 1
    # position = np.dot(position, tform.params.T)
    # position[:, 2] = image_vertices[:, 2]*tform.params[0, 0] # scale z
    # position[:, 2] = position[:, 2] - np.min(position[:, 2]) # translate z

    # 4. uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c = 3)
    
    # show_mesh(image_path, vertices, bfm.kpt_ind)
    kpt = uv_position_map[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    show_uv_mesh(image_path, uv_position_map, kpt)
    # 5. save files
    # io.imsave('{}/{}'.format(save_folder, image_name), np.squeeze(cropped_image))
    # np.save('{}/{}'.format(save_uv_folder, image_name.replace('jpg', 'npy')), uv_position_map)
    # io.imsave('{}/{}'.format(save_img_folder, image_name), img)

    # --verify
    # import cv2
    # uv_texture_map_rec = cv2.remap(cropped_image, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_tex.jpg')), np.squeeze(uv_texture_map_rec))

if __name__ == '__main__':
    save_uv_folder  = os.path.join("/media/viet/Vincent/SPRNet", "train_uv_256x256")
    save_img_folder = os.path.join("/media/viet/Vincent/SPRNet", "train_im_256x256")
    if not os.path.exists(save_uv_folder):
        os.mkdir(save_uv_folder)

    if not os.path.exists(save_img_folder):
        os.mkdir(save_img_folder)

    # set para
    uv_h = uv_w = 256
    image_h  = image_w = 256
    
    # load uv coords
    global uv_coords
    uv_coords = face3d.morphable_model.load.load_uv_coords('/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/BFM/Out/BFM_UV.mat') #
    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    
    # load bfm 
    bfm = MorphabelModel('/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/BFM/Out/BFM.mat') 

    image_path_List = []
    # data_path = os.path.join(str(os.path.abspath(os.getcwd())), "..")
    data_path = "/home/viet/Projects/Pycharm/SPRNet/data/"
    file_list               = os.path.join(data_path, "..", "train.configs", "train_aug_120x120.list.train")
    img_names_list          = Path(file_list).read_text().strip().split('\n')
    param_fp_gt             ='/home/viet/Projects/Pycharm/SPRNet/train.configs/param_all_norm.pkl'
    param_62d               = pickle.load(open(param_fp_gt,'rb'))
    index = 0
    for img_name in tqdm(img_names_list):
        file_name   = os.path.splitext(img_name)[0]
        image_path  = os.path.join("/home/viet/Data/train_aug_120x120", file_name + ".jpg")
        param       = param_62d[index]

        run_posmap_300W_LP(bfm, image_path, param, save_uv_folder, save_img_folder)
        index       = index + 1
    # for img in glob.glob(os.path.join(data_path, "shape", "vertex_gt", "*"), recursive=True):
    #     file_name = os.path.basename(os.path.splitext(img)[0])
    #     print(file_name)
    #     image_path  = os.path.join(data_path, "train_aug_120x120", file_name + ".jpg")
    #     mat_path    = os.path.join(data_path, "shape", "vertex_gt", file_name + ".mat")
    #     run_posmap_300W_LP(bfm, image_path, mat_path, save_folder)
    # run
