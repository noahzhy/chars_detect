import os
import time
import glob
import random
import argparse

import cv2
import onnx
import torch
import numpy as np
from utils.tool import *
from onnxsim import simplify
from module.detector import Detector


yaml = "configs/coco.yaml"
weight = "checkpoint/weight_AP05_0.9946318665641644_180.pth"
img = random.choice(glob.glob("data/val/*.jpg"))
thresh = 0.5
device = torch.device("cpu")
cfg = LoadYaml(yaml)


# cv2_imread to support path
def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


class Demo(object):
    def __init__(self, cfg=cfg, weight=weight, device=device, thresh=thresh):
        self.cfg = cfg
        self.weight = weight
        self.device = device
        self.thresh = thresh

        self.model = Detector(self.cfg.category_num, True).to(self.device)
        self.model.load_state_dict(torch.load(self.weight, map_location=self.device))
        self.model.eval()

    def preprocess(self, img):
        res_img = cv2.resize(img, (self.cfg.input_width, self.cfg.input_height), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, self.cfg.input_height, self.cfg.input_width, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(self.device).float() / 255.0
        return img

    def postprocess(self, preds, thresh):
        output = handle_preds(preds, self.device, thresh)
        return output

    def predict(self, img):
        img = self.preprocess(img)
        preds = self.model(img)
        output = self.postprocess(preds, self.thresh)
        return output


def parse_label(path):
    # split label with _
    labels = os.path.splitext(os.path.basename(path))[0].split('_')
    t_labels = ""

    while len(t_labels) < 7:
        label = labels.pop(0)
        # remove space and _
        label = label.replace(' ', '').replace('_', '')
        t_labels += label

    len_ = len(t_labels)

    # if A-Z in label, len_ += 1, if a-z in label, len_ += 0
    for i in t_labels:
        if i.isupper():
            len_ += 1
            break
        elif i.islower():
            break

    return len_


# move file to dir
def move2dir(file_path, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # get file name
    file_name = os.path.basename(file_path)
    # get file new path
    new_path = os.path.join(dir_path, file_name)
    # move file, if file exists, overwrite it
    if os.path.exists(new_path):
        os.remove(new_path)
    os.rename(file_path, new_path)


# write label to file
def write2file(file_path, label):
    # label from [x1, y1, x2, y2, cls] to [cls, x, y, w, h]
    label = [int(label[4]), (label[0] + label[2]) / 2, (label[1] + label[3]) / 2, label[2] - label[0], label[3] - label[1]]
    # write label to file
    with open(file_path, 'w') as f:
        f.write(' '.join(str(i) for i in label))


if __name__ == '__main__':
    demo = Demo()
    imgs = glob.glob('E:/dataset/LPR_210K/val/*.jpg')

    # 加载label names
    LABEL_NAMES = []
    with open(cfg.names, 'r') as f:
	    for line in f.readlines():
	        LABEL_NAMES.append(line.strip())

    for img in imgs:
        len_ = parse_label(img)
        ori_img = cv2_imread(img)
        output = demo.predict(ori_img)

        H, W, _ = ori_img.shape
        scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

        _pred_labels = []

        for box in output[0]:
            box = box.tolist()
            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]
            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)
            _pred_labels.append(category)

        # to txt
        if len_ == len(_pred_labels):
            write2file(img.replace('jpg', 'txt'), _pred_labels)
            move2dir(img, "E:/dataset/LPR_210K/val")
        else:
            # save log to file
            with open('log.txt', 'a') as f:
                f.write('{}: {} != {}\n'.format(img, len_, len(_pred_labels)))
