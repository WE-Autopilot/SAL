import numpy as np
import torch as pt
from PIL import Image
from model import Model
from get_overlap_distance_2 import get_overlap_distance_2


# map_img is Image object for map, pos is 2d vector, theta is scalar, scale is scalar
def image_lidar(map_img, pos, theta, scale):
    return Image.fromarray(np.random.randn(256, 256))


# img is Image object returned by image_lidar
# maybe just use Model from model.py
def sal(img):
    return pt.randn(32)


# img is Image object returned by image_lidar, pos is 2d vector, theta is scalar, path is 32 long vector, a through d are scalars
def penalty(img, pos, theta, path, a, b, c, d):
    return stage1_penalty(img, pos, theta, path, np.array([0, 0]), a, b, c, d)