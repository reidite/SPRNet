# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pathlib import Path
import numpy as np
import cv2
import glob
from PIL import Image
def show_mesh(img_path, shape_path):
    img = cv2.imread(img_path)

    img = cv2.resize(img, None, fx=4,fy=4,interpolation = cv2.INTER_CUBIC)
    info = sio.loadmat(shape_path)
    x, y, z = info['vertex'].astype(np.float32)
    for i in range(0, x.shape[0], 1):
        img = cv2.circle(img, (int(x[i]) * 4, int(120 - y[i]) * 4), 1, (255, 0, 130 - z[i]), -1)
    cv2.imshow("scatter_face",img)
    cv2.waitKey()

if __name__ == "__main__":
    file_path = (str(os.path.abspath(os.getcwd())))
    data_path = "/home/viet/Data/train_aug_120x120"
    data_list_val = os.path.join(file_path, "train.configs", "train_aug_120x120.list.train")
    img_names_list = Path(data_list_val).read_text().strip().split('\n')
    # data_index = 164600
    for data_index in range(len(img_names_list)):
        file_name   = os.path.splitext(img_names_list[data_index])[0]
        img_path = os.path.join(data_path, file_name + ".jpg")
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=4,fy=4,interpolation = cv2.INTER_CUBIC)
        cv2.imshow("scatter_face",img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    # show_mesh(os.path.join(file_path, "data", "train_aug_120x120", file_name + ".jpg"), os.path.join(file_path, "data", "shape", "vertex_gt", file_name + ".mat"))