import pandas as pd
import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h



# Make heatmaps using the utility functions from the centernet repo
def draw_msra_gaussian(heatmap, center, sigma=2):
  tmp_size = sigma * 6
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap


def make_hm_regr(center, input_size, model_scale, in_scale):
    hm = np.zeros([input_size // model_scale, input_size // model_scale])
    regr = np.zeros([2, input_size // model_scale, input_size // model_scale])
    
    if len(center) == 0:
        return hm, regr
    
    center = np.array(center)
    for c in center:
        sigma = np.clip(c[2] * c[3] // 2000, 2, 4)
        hm = draw_msra_gaussian(hm, 
                                [int(c[0]) // model_scale // in_scale, 
                                 int(c[1]) // model_scale // in_scale], 
                                sigma=sigma)
    
    regrs = center[:, 2:] / (input_size * in_scale)
    for r, c in zip(regrs, center):
        x, y = int(c[0]) // model_scale // in_scale, int(c[1]) // model_scale // in_scale
        regr[:, max(0, x-2):x+3, max(0, y-2):y+3] = np.array([r[0], r[1]]).reshape(2, 1, 1)
    return hm, regr


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, input_size, model_scale, in_scale, folder_path,transform=None):
        self.folder_path= folder_path
        self.transform = transform
        
        self.classes = list()
        self.bboxes = list()
        self.images_path = list()
        self.input_size = input_size
        self.model_scale = model_scale
        self.in_scale = in_scale
        

        file_names = [file_name.replace(".txt", "") for file_name in sorted(
            os.listdir(self.folder_path)) if file_name.endswith(".txt")]
        for file_name in tqdm(file_names):
            bbox = list()
            class_ids = list()
            file_path = os.path.join(self.folder_path, file_name + ".txt")
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    data = [float(x) for x in line.split(" ") if len(line.replace(" ", "")) > 0]
                    if int(data[0]) == 1:
                        box = [i*1024 for i in data[1:]]
                        bbox.append(box)
                        class_ids.append([int(data[0])])

            file_path = os.path.join(self.folder_path, file_name + ".jpg").replace(os.sep, "/")

            self.bboxes.append(bbox)
            self.classes.append(class_ids)
            self.images_path.append(file_path)
        
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self,idx):
        img = cv2.imread(self.images_path[idx])
        if self.transform is not None:
            img = self.transform(img)
        boxes = self.bboxes[idx]
        classes = self.classes[idx]
        classes = np.array(classes)
        hm, regr = make_hm_regr(boxes, self.input_size, self.model_scale, self.in_scale)
        return img, hm, regr
    