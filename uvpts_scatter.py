# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from pathlib import Path
import numpy as np
import cv2

def show_uv_mesh(img_path, uv, keypoint):
    img = cv2.imread(img_path)
    x, y, z = uv.transpose(2, 0, 1).reshape(3, -1)
    img = cv2.resize(img, None, fx=32/15,fy=32/15,interpolation = cv2.INTER_CUBIC)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 1, (255, 0, 0), -1)
    x, y, z = keypoint.transpose().astype(np.int32)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]), int(y[i])), 4, (255, 255, 255), -1)
    # res = cv2.resize(img, None, fx=3,fy=3,interpolation = cv2.INTER_CUBIC)
    cv2.imshow("uv_point_scatter",img)
    cv2.waitKey()

if __name__ == "__main__":
    file_path = (str(os.path.abspath(os.getcwd())))
    data_list_val = os.path.join(file_path, "train.configs", "train_aug_120x120.list.train")
    img_names_list = Path(data_list_val).read_text().strip().split('\n')[120:130]
    # data_index = 200000
    # file_name   = os.path.splitext(img_names_list[data_index])[0]
    # uv_kpt_ind = np.loadtxt("/home/viet/Projects/Pycharm/SPRNet/data/processing/Data/UV/uv_kpt_ind.txt").astype(np.int32)
    # uv_position_map = np.load(os.path.join(file_path, "data", "train_uv_256x256", file_name + ".npy")).astype(np.float32)
    # kpt = uv_position_map[uv_kpt_ind[1,:].astype(np.int32), uv_kpt_ind[0,:].astype(np.int32), :]
    # show_uv_mesh(os.path.join(file_path, "data", "train_aug_120x120", file_name + ".jpg"), uv_position_map, kpt)
    for img_name in tqdm(img_names_list):
        file_name   = os.path.splitext(img_name)[0]
        cmd         = "cp " + str(os.path.join(file_path, "data", "train_uv_256x256", file_name + ".npy")) + " " + str(os.path.join(file_path, "test_rotation", "uv_256x256", file_name + ".npy"))
        os.system(cmd)
        cmd         = "cp " + str(os.path.join(file_path, "data", "train_aug_120x120", file_name + ".jpg")) + " " + str(os.path.join(file_path, "test_rotation", "aug_120x120", file_name + ".jpg"))
        os.system(cmd)