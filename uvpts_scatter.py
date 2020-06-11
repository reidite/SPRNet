# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import os
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import scipy.io as sio
from pathlib import Path
import numpy as np
from utils.toolkit import *
import cv2
import math

def transform_vertices(R, vts):
    p = np.copy(vts).T
    t = np.mean(p, axis=1).reshape(3, 1)
    pc = p - t
    vts = np.linalg.inv(R).dot(pc)
    angle = -80
    rz = np.array([ [math.cos(angle),               0,   math.sin(angle)],     
                    [               0,              1,                  0], 
                    [ -math.sin(angle),              0,   math.cos(angle)]])
    vts = rz.dot(vts)
    vts = vts + t
    return vts.T

if __name__ == "__main__":
    file_path           = (str(os.path.abspath(os.getcwd())))
    data_list_val       = os.path.join(file_path, "test.configs", "AFLW2000-3D.list")
    img_names_list      = Path(data_list_val).read_text().strip().split('\n')
    data_index          = 123
    file_name           = os.path.splitext(img_names_list[data_index])[0]
    uv_position_map     = np.load(os.path.join(file_path, "data", "verify_uv_256x256", file_name + ".npy")).astype(np.float32)
    kpt                 = get_landmarks(uv_position_map)
    img                 = cv2.imread(os.path.join(file_path, "data", "verify_im_256x256", file_name + ".jpg"))
    # show_uv_mesh(img, uv_position_map, kpt, False)

    ### Create Large Pose
    # img                 = img/255.0
    texture             = cv2.remap(img, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    vts                 = get_vertices(uv_position_map)
    clr                 = get_vertices(texture)

    camera_matrix, pose, (s, R, t) = estimate_pose(vts)
    front_vts               = transform_vertices(R, vts)
    img_show    = img
    # img_show                = cv2.resize(img, None, fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    for i in range(0, front_vts.shape[0], 1):
        R = int(clr[i][0])
        G = int(clr[i][1])
        B = int(clr[i][2])
        img_show = cv2.circle(img_show, (int(front_vts[i][0]), int(front_vts[i][1])), 1, (R, G, B), -1)
    cv2.imshow("uv_point_scatter",img_show)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("Success")