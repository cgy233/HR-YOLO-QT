import os
import cv2
import time
import torch
import argparse

import json
import numpy as np

from models.defaults import _C as cfg
from models.hrnet import HighResolutionNet
from utils.transform import keypoints_from_heatmaps, get_transform
from utils.hr_utils.utils import show_image

# checkpoint = torch.load("./checkpoint.pth", map_location="cpu")
# checkpoint = torch.load("./logs/coco_checkpoint.pth", map_location=lambda storage, loc: storage.cuda(0))
checkpoint = torch.load("./logs/coco_checkpoint.pth", map_location={'cuda:1': 'cuda:0'})
model = HighResolutionNet(cfg).cuda()
model.eval()
model.load_state_dict(checkpoint["state_dict"])

data_dir = r"D:\Project\CHOPIN\HRNet\data"

# json_file = "/home/xiang/Data/hand/onehand10k/onehand10k_test.json"
# with open(json_file, "rb") as f:
#     data = json.load(f)
#
# imgnames = data["imgname"]
# boxs = data["box"]
# kpts = data["kpt"]


with torch.no_grad():
    # for i in range(len(imgnames)):
    # image = cv2.imread(os.path.join(data_dir, imgnames[i]))[:,:,::-1]
    image = cv2.imread("./img/hand_box_test.jpg")[:, :, ::-1]
    w, h = image.shape[:-1]
    # box = boxs[i]
    box = [0, 0, w, h]
    x, y, w, h = box
    scale = max(w, h) / 200. * 1.25
    center = (x + w / 2., y + h / 2.)
    rot = 0
    meta = get_transform(center, scale, (256, 256), rot)
    warp_img = cv2.warpAffine(image,
                              meta[:2], (256, 256),
                              flags=cv2.INTER_LINEAR)
    input_tensor = np.transpose(warp_img / 255., (2, 0, 1))
    input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).cuda()
    out = model(input_tensor)
    pred, maxvals = keypoints_from_heatmaps(out.cpu().numpy(),
                                            meta[None])
    # maxvals[maxvals<0.15] = 0
    pred = np.concatenate((pred, maxvals), axis=2)
    # show_image(image, pred[0], "tmp/" + imgnames[i][:-4] + "_coco.jpg")
    show_image(image, pred[0], "./test.jpg")
    # img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
