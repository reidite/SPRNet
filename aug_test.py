import numpy as np
from skimage import io, transform
import math
import copy
from PIL import ImageEnhance, ImageOps, ImageFile, Image
import cv2
from pathlib import Path
import os 
def gaussNoise(x, mean=0, var=0.004):
    noise = np.random.normal(mean, var ** 0.5, x.shape)
    out = x + noise
    out = np.clip(out, 0., 1.0)
    return out

def randomColor(image):
    """
    """
    PIL_image = Image.fromarray((image * 255.).astype(np.uint8))
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(PIL_image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(0, 31) / 10.
    out = np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))
    out = out / 255.
    return out

if __name__ == "__main__":
    file_path = (str(os.path.abspath(os.getcwd())))
    data_list_val = os.path.join(file_path, "train.configs", "train_aug_120x120.list.train")
    img_names_list = Path(data_list_val).read_text().strip().split('\n')[:100]
    for data in img_names_list:
        file_name   = os.path.splitext(data)[0]
        img_path    = os.path.join(file_path, "data", "train_im_256x256", file_name + ".jpg")
        img         = cv2.imread(img_path)
        img_norm    = (img / 255.0).astype(np.float32)
        var = np.random.uniform(0.001, 0.0025)
        print(var)
        img_norm    = gaussNoise(img_norm, 0, var)
        result      = (img_norm * 255.0).astype(np.uint8)
        cv2.imshow("origin",img)
        cv2.imshow("result",result)
        cv2.waitKey()