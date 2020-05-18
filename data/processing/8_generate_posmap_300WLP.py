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
import cv2


import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

uv_kpt_ind = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/uv_kpt_ind.txt").astype(np.int32)
face_ind = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/face_ind.txt").astype(np.int32)
triangles = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/triangles.txt").astype(np.int32)


def process_uv(uv_coords, uv_h = 120, uv_w = 120):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def show_mesh(img_path, vertices, keypoint_index):
    img = cv2.imread(img_path)

    img = cv2.resize(img, None, fx=4,fy=4,interpolation = cv2.INTER_CUBIC)
    x, y, z = vertices
    for i in range(0, x.shape[0], 1):
        if i in keypoint_index:
            img = cv2.circle(img, (int(x[i]) * 4, int(120 - y[i]) * 4), 5, (255, 255, 255), -1)
        else:
            img = cv2.circle(img, (int(x[i]) * 4, int(120 - y[i]) * 4), 1, (255, 0, 130 - z[i]), -1)
    cv2.imshow("3d_point_scatter",img)
    cv2.waitKey()

def show_uv_mesh(img_path, uv, keypoint):
    img = cv2.imread(img_path)
    x, y, z = uv.transpose(2, 0, 1).reshape(3, -1)
    img = cv2.resize(img, None, fx=4,fy=4,interpolation = cv2.INTER_CUBIC)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]) * 4, int(y[i]) * 4), 1, (255, 0, 0), -1)
    x, y, z = keypoint.transpose().astype(np.int32)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]) * 4, int(y[i]) * 4), 5, (255, 255, 255), -1)
    # res = cv2.resize(img, None, fx=3,fy=3,interpolation = cv2.INTER_CUBIC)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()

def run_posmap_300W_LP(bfm, image_path, mat_path, save_folder,  uv_h = 120, uv_w = 120, image_h = 120, image_w = 120):
    # 1. load image and fitted parameters
    image_name = image_path.strip().split('/')[-1]
    image = io.imread(image_path)/255.
    [h, w, c] = image.shape

    info = sio.loadmat(mat_path)
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
    # transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    # projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection as in 3DDFA

    vertices = info['vertex'].astype(np.float32)
    projected_vertices = vertices.transpose()
    image_vertices = projected_vertices.copy()
    image_vertices[:,1] = h - image_vertices[:,1] - 1

    # 3. crop image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top)/2
    size = int(old_size*1.5)
    # random pertube. you can change the numbers
    marg = old_size*0.1
    t_x = np.random.rand()*marg*2 - marg
    t_y = np.random.rand()*marg*2 - marg
    center[0] = center[0]+t_x; center[1] = center[1]+t_y
    size = size*(np.random.rand()*0.2 + 0.9)

    # crop and record the transform parameters
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))


    # transform face position(image vertices) along with 2d facial image 
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2]*tform.params[0, 0] # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2]) # translate z

    # 4. uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c = 3)
    
    # show_mesh(image_path, vertices, bfm.kpt_ind)
    # uv_kpt_ind_120 = np.rint((uv_kpt_ind / 256 * 120))
    # kpt = uv_position_map[uv_kpt_ind_120[1,:].astype(np.int32), uv_kpt_ind_120[0,:].astype(np.int32), :]
    # show_uv_mesh(image_path, uv_position_map, kpt)
    # 5. save files
    # io.imsave('{}/{}'.format(save_folder, image_name), np.squeeze(cropped_image))
    np.save('{}/{}'.format(save_folder, image_name.replace('jpg', 'npy')), uv_position_map)
    # io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_posmap.jpg')), ((uv_position_map)/max(image_h, image_w)).clip(-1, 1)) # only for show

    # --verify
    # import cv2
    # uv_texture_map_rec = cv2.remap(cropped_image, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_tex.jpg')), np.squeeze(uv_texture_map_rec))

if __name__ == '__main__':
    save_folder = os.path.join(str(os.path.abspath(os.getcwd())), "..", "uv")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # set para
    uv_h = uv_w = 120
    image_h  = image_w = 120
    
    # load uv coords
    global uv_coords
    uv_coords = face3d.morphable_model.load.load_uv_coords('/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/BFM/Out/BFM_UV.mat') #
    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    
    # load bfm 
    bfm = MorphabelModel('/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/BFM/Out/BFM.mat') 

    image_path_List = []
    # data_path = os.path.join(str(os.path.abspath(os.getcwd())), "..")
    data_path = "/home/viet/Projects/Pycharm/SPRNet/data/"
    file_list = os.path.join(data_path, "..", "train.configs", "train_aug_120x120.list.train")
    img_names_list = Path(file_list).read_text().strip().split('\n')[150000:200000]
    #miss 150000 200000
    for img_name in tqdm(img_names_list):
        file_name   = os.path.splitext(img_name)[0]
        image_path  = os.path.join(data_path, "train_aug_120x120", file_name + ".jpg")
        mat_path    = os.path.join(data_path, "shape", "vertex_gt", file_name + ".mat")
        run_posmap_300W_LP(bfm, image_path, mat_path, save_folder)
    # for img in glob.glob(os.path.join(data_path, "shape", "vertex_gt", "*"), recursive=True):
    #     file_name = os.path.basename(os.path.splitext(img)[0])
    #     print(file_name)
    #     image_path  = os.path.join(data_path, "train_aug_120x120", file_name + ".jpg")
    #     mat_path    = os.path.join(data_path, "shape", "vertex_gt", file_name + ".mat")
    #     run_posmap_300W_LP(bfm, image_path, mat_path, save_folder)
    # run
    
